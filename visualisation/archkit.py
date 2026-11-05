"""Description d'architecture pour le visualiseur (skill thibault-model-viz).

Deux etages :

1. `Probe` mesure le vrai modele. Un forward sur une entree factice, un hook sur
   chaque module, et on retient pour chacun les formes d'entree et de sortie, le
   nombre de parametres et le nombre d'appels. Rien n'est recopie a la main, donc
   les dimensions affichees sont celles du code.

2. `node()` construit le schema lisible, celui qu'on dessinerait dans un papier.
   Un bloc peut pointer vers un module (`module="blocks.3.attn"`). C'est ce lien
   qui lui apporte ses parametres, ses formes, et ses metriques d'entrainement
   (`layer/<module>/...` dans MLflow, voir layer_stats.py).

`write()` assemble le tout dans visualisation/arch/<nom>.json et verifie la
couverture : tout module a parametres que le schema ne couvre pas est liste.
C'est le garde-fou contre le bloc oublie.

Le format complet (types de blocs, dispositions, liens) est decrit dans
references/arch-spec.md du skill.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

# Blocs dessines comme de petits ronds d'operation plutot que comme des cartes.
OPS = {"add", "mul", "concat", "split"}

KINDS = {
    "input", "output", "embed", "pos", "norm", "attention", "linear", "rope",
    "sdpa", "softmax", "act", "ffn", "moe", "router", "expert", "head", "loss",
    "mtp", "block", "stack", "cache", "latent", "group", "op", *OPS,
}


# --- mesure du modele ------------------------------------------------------


class Probe:
    """Formes, parametres et appels de chaque module, mesures sur un forward.

    Args:
        model: le module a mesurer, en mode eval de preference.
        run: fonction sans argument qui lance le forward, par exemple
            ``lambda: model(idx, targets)``.
        symbols: dimensions a remplacer par un nom dans l'affichage, par
            exemple ``{"B": 5, "T": 37}``. Choisir des valeurs qui ne
            ressemblent a aucune vraie dimension du modele (pas 2, 3, 8, 64).
    """

    def __init__(
        self,
        model: nn.Module,
        run: Callable[[], Any],
        *,
        symbols: dict[str, int] | None = None,
    ) -> None:
        self.model = model
        self.symbols = symbols or {}
        self._names = self._symbol_table(self.symbols)
        self.calls: dict[str, int] = {}
        self.shapes: dict[str, tuple[str | None, str | None]] = {}
        self.classes = {path: type(m).__name__ for path, m in model.named_modules()}

        handles = [
            module.register_forward_hook(self._hook(path))
            for path, module in model.named_modules()
        ]
        try:
            with torch.no_grad():
                run()
        finally:
            for handle in handles:
                handle.remove()

        # nom canonique de chaque parametre : un poids partage n'est compte
        # qu'une fois, sous le premier module qui le declare
        self.param_owner = {
            name: p.numel() for name, p in model.named_parameters()
        }
        self.total = sum(self.param_owner.values())

    @staticmethod
    def _symbol_table(symbols: dict[str, int]) -> dict[int, str]:
        names: dict[int, str] = {}
        b, t = symbols.get("B"), symbols.get("T")
        if b and t:
            names[b * t] = "B·T"
        if t:
            for k in (1, 2, 3):
                names.setdefault(t - k, f"T−{k}")
        for name, value in symbols.items():
            names[value] = name
        return names

    def _fmt(self, value: Any) -> str | None:
        if torch.is_tensor(value):
            dims = ", ".join(self._names.get(int(d), str(int(d))) for d in value.shape)
            return f"[{dims}]"
        if isinstance(value, (tuple, list)):
            for item in value:
                text = self._fmt(item)
                if text is not None:
                    return text
        return None

    def _hook(self, path: str):
        def hook(module, inputs, output):
            self.calls[path] = self.calls.get(path, 0) + 1
            if path not in self.shapes:
                self.shapes[path] = (self._fmt(inputs), self._fmt(output))

        return hook

    # --- lecture -----------------------------------------------------------

    def has(self, path: str) -> bool:
        return path in self.classes

    def params(self, path: str) -> int:
        """Parametres possedes par le module et ses enfants (poids partages exclus)."""
        if not path:
            return self.total
        prefix = path + "."
        return sum(n for name, n in self.param_owner.items() if name.startswith(prefix))

    def inp(self, path: str) -> str | None:
        return self.shapes.get(path, (None, None))[0]

    def out(self, path: str) -> str | None:
        return self.shapes.get(path, (None, None))[1]


# --- construction du schema ------------------------------------------------


def node(
    label: str,
    kind: str = "block",
    *,
    module: str | None = None,
    info: list[str] | tuple[str, ...] = (),
    children: list[dict] | tuple[dict, ...] = (),
    layout: str = "col",
    flow: str = "seq",
    box: bool = True,
    skip: bool | str | None = None,
    cols: int | None = None,
    shape: str | None = None,
    doc: str | None = None,
    ref: str | None = None,
    badge: str | None = None,
    scale: float | None = None,
    id: str | None = None,
) -> dict:
    """Un bloc du schema.

    Args:
        label: titre affiche.
        kind: famille du bloc, fixe la couleur et le picto (voir KINDS).
        module: chemin `named_modules()` du module que ce bloc represente.
        info: lignes courtes sous le titre (dimensions, tetes, d_ff...).
        children: sous-blocs, visibles en zoomant.
        layout: ``col`` (haut vers bas), ``row`` (gauche vers droite) ou ``grid``.
        flow: ``seq`` (fleches entre enfants), ``par`` (branches paralleles)
            ou ``none`` (aucune fleche, pour une grille d'experts).
        box: False pour un simple regroupement sans cadre ni titre.
        skip: connexion residuelle, du debut du groupe vers le dernier enfant
            (True) ou vers l'enfant dont l'id est donne.
        cols: colonnes d'une grille.
        shape: forme de sortie a afficher si le module ne la donne pas.
        doc: explication affichee dans le panneau de detail.
        ref: reference papier (``arXiv 2305.13245``).
        badge: pastille courte sur la carte (``×12``, ``top-6``).
        scale: agrandit une carte simple, pour les blocs du chemin principal
            qui doivent rester lisibles dans la vue d'ensemble (3 a 5).
        id: identifiant global, utile pour les liens. Sinon derive du chemin.
    """
    if kind not in KINDS:
        raise ValueError(f"kind inconnu {kind!r}, choisir parmi {sorted(KINDS)}")
    data = {
        "label": label, "kind": kind, "module": module, "info": list(info),
        "children": list(children), "layout": layout, "flow": flow, "box": box,
        "skip": skip, "cols": cols, "shape": shape, "doc": doc, "ref": ref,
        "badge": badge, "scale": scale, "id": id,
    }
    defaults = {"layout": "col", "flow": "seq", "box": True}
    return {
        k: v for k, v in data.items()
        if v not in (None, [], "") and defaults.get(k, object()) != v
    }


def link(
    src: str, dst: str, label: str = "", *, style: str = "dashed", kind: str = "group"
) -> dict:
    """Lien hors du flux principal (poids partages, cross-attention, MTP)."""
    return {"from": src, "to": dst, "label": label, "style": style, "kind": kind}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "n"


def _assign_ids(root: dict) -> list[dict]:
    """Donne un id unique a chaque bloc, et renvoie la liste a plat."""
    used: set[str] = set()
    flat: list[dict] = []

    def visit(n: dict, parent_id: str) -> None:
        if "id" in n:
            nid = n["id"]
            if nid in used:
                raise ValueError(f"id en double dans le schema : {nid!r}")
        else:
            base = f"{parent_id}/{_slug(n['label'])}" if parent_id else _slug(n["label"])
            nid, i = base, 2
            while nid in used:
                nid, i = f"{base}-{i}", i + 1
            n["id"] = nid
        used.add(nid)
        flat.append(n)
        for child in n.get("children", []):
            visit(child, nid)

    visit(root, "")
    return flat


def write(
    path: str | Path,
    root: dict,
    probe: Probe,
    *,
    name: str,
    title: str,
    subtitle: str = "",
    summary: list[tuple[str, str]] | None = None,
    links: list[dict] | None = None,
    source: str = "",
) -> dict:
    """Complete le schema avec les mesures, verifie la couverture, ecrit le JSON."""
    flat = _assign_ids(root)
    ids = {n["id"] for n in flat}
    for lk in links or []:
        for end in (lk["from"], lk["to"]):
            if end not in ids:
                raise ValueError(f"lien vers un id absent du schema : {end!r}")

    unknown = []
    for n in flat:
        mod = n.get("module")
        if mod is None:
            continue
        if not probe.has(mod):
            unknown.append(mod)
            continue
        n["params"] = probe.params(mod)
        n["cls"] = probe.classes.get(mod)
        n["calls"] = probe.calls.get(mod, 0)
        n.setdefault("in", probe.inp(mod))
        if "shape" in n:
            n["out"] = n.pop("shape")
        else:
            n.setdefault("out", probe.out(mod))
        for key in ("in", "out"):
            if n.get(key) is None:
                n.pop(key, None)
    for n in flat:
        if "shape" in n:
            n["out"] = n.pop("shape")

    # La couverture ne compte que les blocs du niveau le plus fin : un conteneur
    # lie a "blocks.3" ne doit pas cacher une matrice oubliee a l'interieur.
    def has_bound_child(n: dict) -> bool:
        return any(c.get("module") or has_bound_child(c) for c in n.get("children", []))

    bound = sorted({n["module"] for n in flat if n.get("module") and not has_bound_child(n)})
    missing: dict[str, int] = {}
    covered = 0
    for pname, numel in probe.param_owner.items():
        if any(pname.startswith(m + ".") for m in bound):
            covered += numel
        else:
            owner = pname.rsplit(".", 1)[0]
            missing[owner] = missing.get(owner, 0) + numel

    # Modules a poids que le forward de mesure n'appelle jamais : code mort, option
    # ignoree par une variante, ou poids lus directement (a verifier a la main).
    # Les experts MoE sont exclus, un expert peut ne recevoir aucun token par hasard.
    never: dict[str, int] = {}
    for mpath, mod in probe.model.named_modules():
        if probe.calls.get(mpath, 0) or not any(True for _ in mod.parameters(recurse=False)):
            continue
        if any(part in ("experts", "shared_experts", "local_experts") for part in mpath.split(".")):
            continue
        role = re.sub(r"\.\d+\.", ".#.", f".{mpath}.").strip(".")
        never[role] = never.get(role, 0) + 1

    data = {
        "name": name,
        "title": title,
        "subtitle": subtitle,
        "source": source,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "symbols": probe.symbols,
        "params_total": probe.total,
        "summary": [list(item) for item in (summary or [])],
        "coverage": {
            "covered": covered,
            "total": probe.total,
            "missing": [{"module": m, "params": n} for m, n in sorted(missing.items())],
            "unknown_modules": unknown,
            "never_called": [{"module": m, "count": c} for m, c in sorted(never.items())],
        },
        "links": links or [],
        "root": root,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    pct = 100 * covered / probe.total if probe.total else 100.0
    print(f"[archkit] {out}  ({len(flat)} blocs, {len(links or [])} liens)")
    # trois decimales des qu'il manque un module, sinon 99,998 % s'afficherait 100.0 %
    shown = f"{pct:.1f}" if not missing else f"{min(pct, 99.999):.3f}"
    print(f"[archkit] couverture : {shown} % des {probe.total:,} parametres")
    for m, n in sorted(missing.items(), key=lambda kv: -kv[1])[:15]:
        print(f"[archkit]   non couvert : {m}  ({n:,} params)")
    for m in unknown:
        print(f"[archkit]   module introuvable : {m}")
    for m, c in sorted(never.items()):
        print(f"[archkit]   jamais appele pendant le forward : {m}  (x{c})")
    return data
