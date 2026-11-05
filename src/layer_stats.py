"""Mesures par couche pour le visualiseur d'architecture (skill thibault-model-viz).

Le visualiseur colore chaque bloc du schema avec une mesure de l'entrainement :
norme des gradients, ratio de mise a jour, RMS des activations, charge des
experts MoE. Ces valeurs passent par MLflow comme toutes les autres metriques.
Ce module les calcule, `RunLogger.metric` les envoie.

Convention de nommage, celle que lit le visualiseur :

    layer/<chemin du module>/<stat>     ex. layer/blocks.3.attn.sublayer.q_proj/grad_norm

Le chemin est celui de `model.named_modules()`, donc le schema retrouve tout seul
le bloc correspondant. Stats envoyees :

    grad_norm     norme L2 des gradients du module, avant clipping
    weight_norm   norme L2 des poids
    update_ratio  ||w_apres - w_avant|| / ||w_avant|| sur le step mesure
    act_rms       RMS de la sortie du module, moyenne sur les micro-batchs
    tokens        lignes recues par un expert MoE (la charge s'en deduit)

Usage, autour de l'optimizer :

    stats = LayerStats(model, every=100)
    run.log(stats.describe())
    for step in run.track(range(max_steps)):
        stats.start(step)                  # arme les hooks si le step est mesure
        ...forward, backward...
        scaler.unscale_(optimizer)         # gradients en vraie grandeur (AMP fp16)
        stats.before_optimizer()           # normes des gradients, copie des poids
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        run.metric(step=step, **stats.collect())   # {} hors des steps mesures

Cout : rien hors des steps mesures, les hooks ressortent tout de suite. Sur un
step mesure, une copie des poids (coupee par updates=False) et une seule
synchronisation GPU. Avec torch.compile, passer activations=False : les hooks
de forward forceraient une recompilation.
"""

from __future__ import annotations

import torch
from torch import nn

PREFIX = "layer"

# Listes dont chaque enfant est suivi comme un tout (un expert = une unite).
# Sans ce regroupement, un MoE de 64 experts sortirait 3 matrices x 64 experts
# x 3 stats par couche, illisible et lourd pour le store MLflow.
COLLAPSE = ("experts", "shared_experts", "local_experts")

# Au-dela, la RMS d'une activation est estimee sur un echantillon regulier,
# pour ne jamais copier en fp32 des logits de plusieurs Go.
MAX_RMS_ELEMENTS = 1 << 24


class LayerStats:
    """Calcule les metriques `layer/...` d'un modele PyTorch."""

    def __init__(
        self,
        model: nn.Module,
        every: int = 100,
        *,
        activations: bool = True,
        updates: bool = True,
        collapse: tuple[str, ...] = COLLAPSE,
    ) -> None:
        # un modele passe par torch.compile garde l'original dans _orig_mod
        self.model = getattr(model, "_orig_mod", model)
        self.every = max(int(every), 0)
        self.updates = updates
        self.collapse = collapse
        self.units = self._find_units()
        self.experts = [p for p in self.units if self._is_collapsed_root(p)]

        self._armed = False
        self._grad: dict[str, torch.Tensor] = {}
        self._before: dict[str, list[torch.Tensor]] = {}
        self._act: dict[str, list[torch.Tensor]] = {}
        self._tokens: dict[str, int] = {}
        self._handles: list = []
        self.hooked = 0
        if activations and self.every:
            self._register_hooks()

    # --- decouverte des unites -------------------------------------------

    def _is_collapsed_root(self, path: str) -> bool:
        parts = path.split(".")
        return len(parts) >= 2 and parts[-2] in self.collapse and parts[-1].isdigit()

    def _inside_collapsed(self, path: str) -> bool:
        parts = path.split(".")
        return any(
            parts[i] in self.collapse and parts[i + 1].isdigit()
            for i in range(len(parts) - 2)
        )

    def _find_units(self) -> dict[str, list[nn.Parameter]]:
        """Un module qui porte directement des poids est une unite.

        Les poids partages (embedding lie a la tete) ne comptent qu'une fois,
        sous le premier nom que leur donne PyTorch.
        """
        units: dict[str, list[nn.Parameter]] = {}
        seen: set[int] = set()
        for path, module in self.model.named_modules():
            if self._inside_collapsed(path):
                continue
            recurse = self._is_collapsed_root(path)
            params = [
                p for p in module.parameters(recurse=recurse)
                if p.requires_grad and id(p) not in seen
            ]
            if params:
                seen.update(id(p) for p in params)
                units[path or "_root"] = params
        return units

    # --- hooks d'activation ----------------------------------------------

    def _register_hooks(self) -> None:
        """Hooks sur chaque unite et sur chaque element d'une liste de blocs.

        Les elements de liste (blocks.3, mtp.modules_.0) donnent la RMS du flux
        residuel couche par couche, ce que les matrices seules ne montrent pas.
        """
        for path, module in self.model.named_modules():
            if not path or self._inside_collapsed(path):
                continue
            is_block = path.split(".")[-1].isdigit() and not self._is_collapsed_root(path)
            if path in self.units or is_block:
                handle = module.register_forward_hook(
                    self._make_hook(path, count_tokens=self._is_collapsed_root(path))
                )
                self._handles.append(handle)
        self.hooked = len(self._handles)

    def _make_hook(self, path: str, count_tokens: bool):
        def hook(module, inputs, output):
            if not self._armed:
                return
            out = output[0] if isinstance(output, (tuple, list)) and output else output
            if torch.is_tensor(out) and out.is_floating_point() and out.numel():
                flat = out.detach().reshape(-1)
                if flat.numel() > MAX_RMS_ELEMENTS:
                    flat = flat[:: flat.numel() // MAX_RMS_ELEMENTS + 1]
                self._act.setdefault(path, []).append(flat.float().pow(2).mean())
            if count_tokens and inputs and torch.is_tensor(inputs[0]) and inputs[0].dim():
                x = inputs[0]
                self._tokens[path] = self._tokens.get(path, 0) + x.numel() // x.shape[-1]

        return hook

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # --- cycle d'un step ---------------------------------------------------

    def due(self, step: int) -> bool:
        return bool(self.every) and step % self.every == 0

    def start(self, step: int) -> None:
        """A appeler avant le premier forward du step."""
        self._armed = self.due(step)
        self._grad.clear()
        self._before.clear()
        self._act.clear()
        # un expert jamais choisi pendant le step a une charge nulle, a ecrire quand meme
        self._tokens = dict.fromkeys(self.experts, 0) if self._armed else {}

    @torch.no_grad()
    def before_optimizer(self) -> None:
        """Apres backward (et unscale_ en AMP), avant clipping et optimizer.step."""
        if not self._armed:
            return
        for path, params in self.units.items():
            grads = [p.grad for p in params if p.grad is not None]
            if grads:
                self._grad[path] = _norm(grads)
            if self.updates:
                self._before[path] = [p.detach().clone() for p in params]

    @torch.no_grad()
    def collect(self) -> dict[str, float]:
        """Apres optimizer.step. Renvoie {} hors des steps mesures."""
        if not self._armed:
            return {}
        self._armed = False

        names: list[str] = []
        values: list[torch.Tensor] = []

        def add(path: str, stat: str, value: torch.Tensor) -> None:
            names.append(f"{PREFIX}/{path}/{stat}")
            values.append(value.float().reshape(()))

        for path, params in self.units.items():
            add(path, "weight_norm", _norm(params))
            if path in self._grad:
                add(path, "grad_norm", self._grad[path])
            before = self._before.get(path)
            if before is not None:
                delta = _norm([p.detach() - b for p, b in zip(params, before, strict=True)])
                add(path, "update_ratio", delta / _norm(before).clamp_min(1e-12))
        for path, squares in self._act.items():
            dev = squares[0].device
            add(path, "act_rms", torch.stack([s.to(dev) for s in squares]).mean().sqrt())

        out: dict[str, float] = {}
        if values:
            device = values[0].device
            # une seule synchronisation GPU pour toutes les valeurs
            flat = torch.stack([v.to(device) for v in values]).cpu().tolist()
            out.update(zip(names, flat, strict=True))
        for path, count in self._tokens.items():
            out[f"{PREFIX}/{path}/tokens"] = float(count)

        self._grad.clear()
        self._before.clear()
        self._act.clear()
        return out

    def describe(self) -> str:
        """Une ligne pour run.log."""
        if not self.every:
            return "layers   : mesures par couche desactivees"
        experts = f", {len(self.experts)} experts" if self.experts else ""
        return (
            f"layers   : {len(self.units)} unites{experts}, {self.hooked} hooks, "
            f"tous les {self.every} steps (layer/... dans MLflow)"
        )


def _norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Norme L2 globale d'une liste de tenseurs, accumulee en fp32."""
    parts = [torch.linalg.vector_norm(t.detach().float()) for t in tensors]
    device = parts[0].device
    return torch.linalg.vector_norm(torch.stack([p.to(device) for p in parts]))
