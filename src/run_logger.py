"""Journalisation des entrainements IA de Thibault GAREL.

Deux objectifs, un seul objet a instancier :

1. Un `run.log` qui se suit **en le lisant**, sans outil ni filtre. Lignes
   horodatees, environnement rappele en tete, metriques alignees en colonnes,
   resume final. Aucune frame tqdm ne pollue le fichier quand la sortie est
   redirigee (nohup, `Start-Process`), sinon 46 lignes utiles pesent 878 Ko.

2. Tout dans MLflow, qui porte les courbes. Params, metriques, artefacts et
   statut du run sont envoyes sans un seul appel MLflow a ecrire.

Les fichiers atterrissent dans la structure standard du projet :

    <racine projet>/
    ├── outputs/
    │   ├── logs/<model>_run-<NN>/       run.log, metrics.jsonl, config.json
    │   ├── models/<model>_run-<NN>_date-<YYYY-MM-DD>/
    │   └── results/<model>_run-<NN>_date-<YYYY-MM-DD>/
    └── mlruns/                          store MLflow, jamais renomme

Usage (depuis la racine du projet, `src/` etant un package) :

    from src.run_logger import RunLogger

    with RunLogger("resnet-18") as run:
        run.config(cfg.model_dump())
        for step, batch in enumerate(run.track(loader, desc="train")):
            run.metric(step=step, **{"loss/ce": loss})
        run.metric(step=step, show=True, **{"loss/val": val})
        torch.save(model.state_dict(), run.model_dir / "best_model.pt")

Aucune dependance obligatoire, le fichier se copie tel quel dans un projet.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

# Marqueurs qui identifient la racine d'un projet, du plus fiable au moins fiable.
ROOT_MARKERS = (".git", "pyproject.toml", "setup.py", "requirements.txt")

# Repli quand le script ne tourne pas dans un projet structure.
FALLBACK_ROOT = Path(r"C:\0-Code_py_temp\0-log_progress")

# Pourcentage entre deux lignes de progression, et largeur de la barre ASCII.
PROGRESS_EVERY_PCT = 5
BAR_WIDTH = 10

# Mesures par couche (skill thibault-model-viz) : envoyees a MLflow comme le reste,
# mais jamais dans la ligne de progression, qui porterait des centaines de valeurs.
LAYER_PREFIX = "layer/"


def find_project_root(start: Path | None = None) -> Path | None:
    """Remonte l'arborescence jusqu'a un marqueur de projet."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in ROOT_MARKERS):
            return candidate
        if (candidate / "outputs").is_dir() and (candidate / "src").is_dir():
            return candidate
    return None


def _next_index(logs_dir: Path, model: str) -> int:
    """Numero de run suivant, deduit des dossiers deja presents."""
    pattern = re.compile(rf"^{re.escape(model)}_run-(\d+)$")
    numbers = [
        int(m.group(1))
        for p in logs_dir.glob(f"{model}_run-*")
        if p.is_dir() and (m := pattern.match(p.name))
    ]
    return max(numbers, default=0) + 1


def _git_state(root: Path) -> str:
    """Commit, branche et proprete de l'arbre, pour rendre le run reproductible."""
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()

    try:
        commit = run("rev-parse", "--short", "HEAD")
        if not commit:
            return "pas un depot git"
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(run("status", "--porcelain"))
        return f"{commit} ({branch}, {'modifie' if dirty else 'propre'})"
    except (OSError, subprocess.SubprocessError):
        return "indisponible"


def _hardware() -> str:
    """Device et VRAM, sans imposer torch aux projets qui ne l'utilisent pas."""
    try:
        import torch
    except ImportError:
        return "torch absent"
    if not torch.cuda.is_available():
        return f"torch {torch.__version__}, cpu"
    props = torch.cuda.get_device_properties(0)
    return (
        f"torch {torch.__version__}, cuda sur {props.name} "
        f"({props.total_memory / 1024**3:.1f} Go)"
    )


class RunLogger:
    """Ecrit le journal du run et alimente MLflow."""

    def __init__(
        self,
        model: str,
        project_root: Path | str | None = None,
        experiment: str | None = None,
        mlflow: bool = True,
        system_metrics: bool = False,
    ):
        self.model = model
        root = Path(project_root) if project_root else find_project_root()
        self.in_project = root is not None
        self.root = root or FALLBACK_ROOT
        self.experiment = experiment or (self.root.name if self.in_project else model)

        logs_parent = (self.root / "outputs" / "logs") if self.in_project else self.root
        logs_parent.mkdir(parents=True, exist_ok=True)
        self.index = _next_index(logs_parent, model)
        self.name = f"{model}_run-{self.index:02d}"
        self.dir = logs_parent / self.name
        self.dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.dir / "run.log"
        self.metrics_path = self.dir / "metrics.jsonl"
        self.config_path = self.dir / "config.json"

        self._log_fh = self.log_path.open("a", encoding="utf-8")
        self._metrics_fh = self.metrics_path.open("a", encoding="utf-8")
        self._start = time.time()
        self._last: dict[str, float] = {}
        self._best: tuple[str, float, int] | None = None
        self._mlflow = None

        self._write_header()
        if mlflow:
            self._start_mlflow(system_metrics)

    # --- dossiers de sortie ----------------------------------------------

    @property
    def _dated(self) -> str:
        return f"{self.name}_date-{datetime.now():%Y-%m-%d}"

    @property
    def model_dir(self) -> Path:
        """outputs/models/<model>_run-<NN>_date-<date>/, cree a la demande."""
        path = (self.root / "outputs" / "models" / self._dated) if self.in_project else self.dir / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def results_dir(self) -> Path:
        """outputs/results/<model>_run-<NN>_date-<date>/, cree a la demande."""
        path = (self.root / "outputs" / "results" / self._dated) if self.in_project else self.dir / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # --- journal ---------------------------------------------------------

    def log(self, message: str) -> None:
        """Une ligne horodatee, dans le fichier et sur la sortie standard."""
        line = f"[{datetime.now():%H:%M:%S}] {message}"
        self._log_fh.write(line + "\n")
        self._log_fh.flush()
        print(line, flush=True)

    def section(self, title: str) -> None:
        self.log(f"--- {title} ---")

    def _write_header(self) -> None:
        """L'en-tete repond aux questions qu'on se pose en relisant un vieux log."""
        self.log(f"=== {self.name} ===")
        self.log(f"projet   : {self.root}")
        self.log(f"python   : {platform.python_version()} | {_hardware()}")
        if self.in_project:
            self.log(f"git      : {_git_state(self.root)}")
        self.log(f"logs     : {self.log_path}")
        if not self.in_project:
            self.log("note     : aucun projet detecte, repli sur 0-log_progress")

    # --- metriques -------------------------------------------------------

    def config(self, data: dict) -> None:
        """Sauve la config et l'envoie en params MLflow."""
        self.config_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        flat = _flatten(data)
        preview = ", ".join(f"{k}={v}" for k, v in list(flat.items())[:6])
        self.log(f"config   : {preview}{' ...' if len(flat) > 6 else ''}")
        self.log(f"           detail complet dans {self.config_path.name}")
        if self._mlflow is not None:
            self._mlflow_safe(self._mlflow.log_params, flat)

    def metric(self, step: int | None = None, show: bool = False, **values: Any) -> None:
        """Envoie les metriques vers MLflow et le metrics.jsonl.

        `show=True` ecrit aussi une ligne dans le journal. Sans lui, les valeurs
        sont retenues et apparaissent dans la prochaine ligne de progression, ce
        qui evite un journal de 500 lignes pour 500 steps.
        """
        row: dict[str, Any] = {}
        if step is not None:
            row["step"] = step
        row["t"] = round(time.time() - self._start, 3)
        row.update(values)
        self._metrics_fh.write(json.dumps(row, default=float) + "\n")
        self._metrics_fh.flush()

        numeric = {k: float(v) for k, v in values.items() if isinstance(v, (int, float))}
        readable = {k: v for k, v in numeric.items() if not k.startswith(LAYER_PREFIX)}
        self._last.update(readable)
        if numeric and self._mlflow is not None:
            self._mlflow_safe(self._mlflow.log_metrics, numeric, step=step)
        if show and readable:
            prefix = f"step {step:>6d}  " if step is not None else ""
            self.log(f"  {prefix}{_format_metrics(readable)}")

    def best(self, name: str, value: float, step: int) -> bool:
        """Retient le meilleur score et signale les ameliorations.

        Renvoie True quand c'est un nouveau record, pour declencher une
        sauvegarde de modele du cote appelant.
        """
        improved = self._best is None or value < self._best[1]
        if improved:
            self._best = (name, value, step)
            self.log(f"  step {step:>6d}  {name} {value:.4f}  (meilleur)")
            if self._mlflow is not None:
                self._mlflow_safe(self._mlflow.log_metric, f"best/{name}", value, step=step)
        return improved

    # --- progression -----------------------------------------------------

    def track(self, iterable: Iterable, desc: str = "train", total: int | None = None) -> Iterator:
        """Barre tqdm si un terminal ecoute, sinon des lignes espacees.

        Sans terminal, la ligne de progression embarque les dernieres metriques
        recues, donc le journal se lit comme un tableau de bord.
        """
        if total is None:
            total = len(iterable) if hasattr(iterable, "__len__") else None

        if _is_tty():
            try:
                from tqdm import tqdm

                yield from tqdm(iterable, desc=desc, total=total)
                return
            except ImportError:
                pass

        yield from self._track_quiet(iterable, desc, total)

    def _track_quiet(self, iterable: Iterable, desc: str, total: int | None) -> Iterator:
        start = time.time()
        next_pct = 0
        count = 0

        for count, item in enumerate(iterable, start=1):
            yield item
            elapsed = time.time() - start
            if total:
                pct = int(100 * count / total)
                if pct < next_pct:
                    continue
                next_pct = pct + PROGRESS_EVERY_PCT
                rate = count / elapsed if elapsed else 0.0
                eta = (total - count) / rate if rate else 0.0
                filled = round(BAR_WIDTH * count / total)
                bar = "#" * filled + " " * (BAR_WIDTH - filled)
                head = f"{desc} {pct:3d}%|{bar}| {count}/{total} ETA {_hms(eta)}"
            else:
                if count % 100:
                    continue
                head = f"{desc} {count} iterations en {_hms(elapsed)}"
            self.log(f"  {head}   {_format_metrics(self._last)}".rstrip())

        self.log(f"  {desc} termine, {count} iterations en {_hms(time.time() - start)}")

    # --- mlflow ----------------------------------------------------------

    def _start_mlflow(self, system_metrics: bool) -> None:
        """Ouvre un run MLflow sans jamais faire echouer l'entrainement."""
        # MLflow >= 3.15 refuse le store fichier ./mlruns sans cette variable
        # ("maintenance mode"). On garde mlruns/ a la racine du projet, qui est
        # la regle du skill thibault-logging et ce qui permet `mlflow ui` sans
        # argument, plutot que de basculer sur une base sqlite. A poser avant
        # l'import, mlflow lit l'environnement au chargement.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        try:
            import mlflow as mlflow_lib
        except ImportError:
            self.log("mlflow   : absent, metriques seulement dans metrics.jsonl")
            return

        try:
            store = self.root / "mlruns"
            store.mkdir(parents=True, exist_ok=True)
            mlflow_lib.set_tracking_uri(store.resolve().as_uri())
            mlflow_lib.set_experiment(self.experiment)
            if system_metrics:
                mlflow_lib.enable_system_metrics_logging()
            mlflow_lib.start_run(run_name=self.name)
            self._mlflow = mlflow_lib
            self.log(f"mlflow   : {store} (experience {self.experiment})")
        except Exception as exc:  # noqa: BLE001
            self.log(f"mlflow   : indisponible ({exc.__class__.__name__}), suivi limite au metrics.jsonl")
            self._mlflow = None

    def _mlflow_safe(self, action, *args, **kwargs) -> None:
        if self._mlflow is None:
            return
        try:
            action(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self.log(f"mlflow desactive apres une erreur : {exc.__class__.__name__}")
            self._mlflow = None

    # --- fin -------------------------------------------------------------

    def done(self, message: str = "termine") -> None:
        self.log(f"--- {message} en {_hms(time.time() - self._start)} ---")
        if self._best:
            name, value, step = self._best
            self.log(f"meilleur : {name} {value:.4f} au step {step}")
        if self._last:
            self.log(f"final    : {_format_metrics(self._last)}")
        models = self.root / "outputs" / "models" / self._dated
        if models.is_dir() and any(models.iterdir()):
            self.log(f"modeles  : {models}")
        if self._mlflow is not None:
            where = f" (depuis {self.root})" if self.in_project else ""
            self.log(f"courbes  : mlflow ui{where}")
        self.close()

    def close(self, status: str = "FINISHED") -> None:
        if self._mlflow is not None:
            self._mlflow_safe(self._mlflow.log_artifact, str(self.log_path))
            if self.config_path.is_file():
                self._mlflow_safe(self._mlflow.log_artifact, str(self.config_path))
            self._mlflow_safe(self._mlflow.end_run, status=status)
            self._mlflow = None
        for fh in (self._log_fh, self._metrics_fh):
            if not fh.closed:
                fh.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.log(f"ECHEC : {exc_type.__name__}: {exc}")
            self.close(status="FAILED")
            return
        self.close()


def _is_tty() -> bool:
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def _format_metrics(values: dict[str, float]) -> str:
    """Metriques compactes et alignees, lisibles dans un log brut."""
    parts = []
    for key, value in values.items():
        if abs(value) >= 1000 or (value and abs(value) < 1e-3):
            parts.append(f"{key} {value:.3e}")
        else:
            parts.append(f"{key} {value:.4f}")
    return "  ".join(parts)


def _flatten(data: dict, prefix: str = "") -> dict[str, str]:
    """Aplatit une config imbriquee, MLflow n'acceptant que des params plats."""
    flat: dict[str, str] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = str(value)[:500]
    return flat


def _hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
