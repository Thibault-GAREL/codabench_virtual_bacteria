"""Serveur local du visualiseur (skill thibault-model-viz).

    python visualisation/server.py                  # depuis la racine du projet
    python visualisation/server.py --port 8766 --no-browser

Sert la page viewer/ et une petite API JSON qui lit :

    visualisation/arch/*.json      les schemas produits par build_arch.py
    mlruns/                        le store MLflow du projet, lu en fichiers
    outputs/logs/<run>/run.log     le journal ecrit par RunLogger

Bibliotheque standard seulement, il tourne donc dans n'importe quel venv, et sur
un pod RunPod derriere un tunnel (`ssh -L 8765:localhost:8765 ...`).

Les metriques sont relues de facon incrementale : un fichier deja lu n'est
repris que depuis sa derniere position. Suivre un run de 24 000 steps en direct
reste donc leger.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
VIEWER = HERE / "viewer"
ARCH_DIR = HERE / "arch"

STATUS = {"1": "RUNNING", "2": "SCHEDULED", "3": "FINISHED", "4": "FAILED", "5": "KILLED"}
LAYER_PREFIX = "layer/"
# un run RUNNING sans nouvelle metrique depuis ce delai est signale comme muet
SILENT_AFTER_S = 600


# --- lecture du store MLflow ----------------------------------------------


def read_meta(path: Path) -> dict[str, str]:
    """meta.yaml de MLflow, des lignes `cle: valeur` a plat."""
    meta: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return meta
    for line in text.splitlines():
        if ":" not in line or line.startswith((" ", "-")):
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip("'\"")
    return meta


def _num(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


class MetricFile:
    """Un fichier de metrique MLflow (`timestamp valeur step` par ligne), lu en continu."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.rest = b""
        self.steps: list[int] = []
        self.values: list[float | None] = []
        self.times: list[int] = []
        self.lock = threading.Lock()

    def refresh(self) -> MetricFile:
        with self.lock:
            try:
                size = self.path.stat().st_size
            except OSError:
                return self
            if size < self.offset:  # fichier reecrit, on repart de zero
                self.__init__(self.path)
            if size == self.offset:
                return self
            with self.path.open("rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
            self.offset += len(chunk)
            data = self.rest + chunk
            lines = data.split(b"\n")
            self.rest = lines.pop()  # ligne pas encore terminee
            for raw in lines:
                parts = raw.decode("utf-8", "replace").split()
                if len(parts) < 2:
                    continue
                step = int(parts[2]) if len(parts) > 2 else len(self.steps)
                self.times.append(int(parts[0]))
                self.values.append(_num(parts[1]))
                self.steps.append(step)
            return self

    def since(self, step: int) -> list[list]:
        return [
            [s, v] for s, v in zip(self.steps, self.values, strict=True) if s > step
        ]


class Store:
    """Le dossier mlruns/ d'un projet."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[Path, MetricFile] = {}
        self.run_dirs: dict[str, Path] = {}
        self.lock = threading.Lock()

    def metric(self, path: Path) -> MetricFile:
        with self.lock:
            mf = self.files.get(path)
            if mf is None:
                mf = self.files[path] = MetricFile(path)
        return mf.refresh()

    def experiments(self) -> list[dict]:
        out = []
        if not self.root.is_dir():
            return out
        for exp_dir in sorted(self.root.iterdir()):
            meta = read_meta(exp_dir / "meta.yaml")
            if not meta or meta.get("lifecycle_stage") == "deleted":
                continue
            runs = [r for r in (self._run(d) for d in exp_dir.iterdir() if d.is_dir()) if r]
            if not runs:
                continue
            runs.sort(key=lambda r: r["start"] or 0, reverse=True)
            out.append({"id": exp_dir.name, "name": meta.get("name", exp_dir.name), "runs": runs})
        out.sort(key=lambda e: max(r["start"] or 0 for r in e["runs"]), reverse=True)
        return out

    def _run(self, run_dir: Path) -> dict | None:
        meta = read_meta(run_dir / "meta.yaml")
        if not meta or meta.get("lifecycle_stage") == "deleted" or "run_id" not in meta:
            return None
        run_id = meta["run_id"]
        self.run_dirs[run_id] = run_dir
        metrics = run_dir / "metrics"
        last = 0.0
        if metrics.is_dir():
            for f in metrics.rglob("*"):
                if f.is_file():
                    last = max(last, f.stat().st_mtime)
        status = STATUS.get(meta.get("status", ""), meta.get("status", "?"))
        silent = status == "RUNNING" and last and time.time() - last > SILENT_AFTER_S
        return {
            "id": run_id,
            "name": meta.get("run_name", run_id[:8]),
            "status": status,
            "silent": bool(silent),
            "start": int(meta["start_time"]) if meta.get("start_time", "").isdigit() else None,
            "end": int(meta["end_time"]) if meta.get("end_time", "").isdigit() else None,
            "last_update": int(last * 1000) if last else None,
            "has_layers": (metrics / "layer").is_dir(),
        }

    def run_dir(self, run_id: str) -> Path | None:
        if run_id not in self.run_dirs:
            self.experiments()
        return self.run_dirs.get(run_id)

    def keys(self, run_id: str, *, layers: bool) -> list[str]:
        base = self.run_dir(run_id)
        if base is None or not (base / "metrics").is_dir():
            return []
        metrics = base / "metrics"
        keys = [f.relative_to(metrics).as_posix() for f in metrics.rglob("*") if f.is_file()]
        if layers:
            return sorted(k for k in keys if k.startswith(LAYER_PREFIX))
        return sorted(k for k in keys if not k.startswith(LAYER_PREFIX))

    def series(self, run_id: str, key: str) -> MetricFile | None:
        base = self.run_dir(run_id)
        if base is None or ".." in key:
            return None
        path = base / "metrics" / key
        return self.metric(path) if path.is_file() else None

    def params(self, run_id: str) -> dict[str, str]:
        base = self.run_dir(run_id)
        if base is None or not (base / "params").is_dir():
            return {}
        params = base / "params"
        out = {}
        for f in sorted(params.rglob("*")):
            if f.is_file():
                try:
                    out[f.relative_to(params).as_posix()] = f.read_text(encoding="utf-8")
                except OSError:
                    pass
        return out


# --- reponses de l'API -----------------------------------------------------


class Api:
    def __init__(self, project: Path) -> None:
        self.project = project
        self.store = Store(project / "mlruns")

    def info(self, q) -> dict:
        return {
            "project": self.project.name,
            "root": str(self.project),
            "mlruns": self.store.root.is_dir(),
            "arch_dir": str(ARCH_DIR),
        }

    def arch_list(self, q) -> list[dict]:
        out = []
        for f in sorted(ARCH_DIR.glob("*.json")):
            try:
                head = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out.append({
                "name": f.stem, "title": head.get("title", f.stem),
                "subtitle": head.get("subtitle", ""), "mtime": int(f.stat().st_mtime * 1000),
            })
        return out

    def arch(self, name: str):
        f = ARCH_DIR / f"{Path(name).name}.json"
        if not f.is_file():
            return None
        return json.loads(f.read_text(encoding="utf-8"))

    def runs(self, q) -> dict:
        return {"experiments": self.store.experiments(), "now": int(time.time() * 1000)}

    def keys(self, run_id: str, q) -> dict:
        out = []
        for key in self.store.keys(run_id, layers=False):
            mf = self.store.series(run_id, key)
            if mf is None or not mf.steps:
                continue
            out.append({
                "key": key, "n": len(mf.steps), "step": mf.steps[-1], "last": mf.values[-1],
            })
        return {"keys": out}

    def metrics(self, run_id: str, q) -> dict:
        keys = [k for k in q.get("keys", [""])[0].split(",") if k]
        since = int(q.get("since", ["-1"])[0])
        out = {}
        for key in keys:
            mf = self.store.series(run_id, key)
            out[key] = mf.since(since) if mf else []
        return {"series": out}

    def status(self, run_id: str, q) -> dict:
        """Avancement d'un run, deduit de la metrique la plus frequente."""
        best = None
        for key in self.store.keys(run_id, layers=False):
            mf = self.store.series(run_id, key)
            if mf and mf.steps and (best is None or len(mf.steps) > len(best.steps)):
                best = mf
        params = self.store.params(run_id)
        max_steps = None
        for name in ("train.max_steps", "max_steps", "train.total_steps", "total_steps"):
            if name in params and params[name].strip().isdigit():
                max_steps = int(params[name])
                break
        out = {"max_steps": max_steps, "step": None, "rate": None, "eta_s": None, "last_ms": None}
        if best is None:
            return out
        out["step"] = best.steps[-1]
        out["last_ms"] = best.times[-1]
        window = min(len(best.steps) - 1, 100)
        if window >= 1:
            dt = (best.times[-1] - best.times[-1 - window]) / 1000
            ds = best.steps[-1] - best.steps[-1 - window]
            if dt > 0 and ds > 0:
                out["rate"] = ds / dt
                if max_steps:
                    out["eta_s"] = max(0.0, (max_steps - 1 - best.steps[-1]) / out["rate"])
        return out

    def layers(self, run_id: str, q) -> dict:
        """Toutes les metriques layer/ alignees sur leurs steps, depuis `since`."""
        since = int(q.get("since", ["-1"])[0])
        values: dict[str, dict[int, float | None]] = {}
        steps: set[int] = set()
        for key in self.store.keys(run_id, layers=True):
            mf = self.store.series(run_id, key)
            if mf is None:
                continue
            col = {s: v for s, v in zip(mf.steps, mf.values, strict=True) if s > since}
            if col:
                values[key[len(LAYER_PREFIX):]] = col
                steps.update(col)
        ordered = sorted(steps)
        return {
            "steps": ordered,
            "values": {k: [col.get(s) for s in ordered] for k, col in values.items()},
        }

    def params(self, run_id: str, q) -> dict:
        return {"params": self.store.params(run_id)}

    def log(self, run_id: str, q) -> dict:
        """Fin du run.log : outputs/logs/<run>/ d'abord, l'artefact MLflow sinon."""
        lines = int(q.get("lines", ["300"])[0])
        base = self.store.run_dir(run_id)
        name = read_meta(base / "meta.yaml").get("run_name", "") if base else ""
        candidates = [self.project / "outputs" / "logs" / name / "run.log"]
        if base is not None:
            candidates.append(base / "artifacts" / "run.log")
        for path in candidates:
            if name and path.is_file():
                return {"path": str(path), "lines": tail(path, lines)}
        return {"path": None, "lines": []}


def tail(path: Path, n: int) -> list[str]:
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        block = min(size, max(4096, n * 200))
        fh.seek(size - block)
        data = fh.read().decode("utf-8", "replace")
    return data.splitlines()[-n:]


# --- serveur HTTP ----------------------------------------------------------


def make_handler(api: Api):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # journal HTTP coupe, il noierait le terminal
            pass

        def _send(self, body: bytes, ctype: str, status=HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status=HTTPStatus.OK) -> None:
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8", status)

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            q = parse_qs(url.query)
            parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
            try:
                if parts[:1] == ["api"]:
                    return self._api(parts[1:], q)
                return self._static(parts)
            except Exception as exc:  # noqa: BLE001
                error = {"error": f"{exc.__class__.__name__}: {exc}"}
                return self._json(error, HTTPStatus.INTERNAL_SERVER_ERROR)

        def _api(self, parts, q):
            if parts == ["info"]:
                return self._json(api.info(q))
            if parts == ["arch"]:
                return self._json(api.arch_list(q))
            if len(parts) == 2 and parts[0] == "arch":
                data = api.arch(parts[1])
                if data:
                    return self._json(data)
                return self._json({"error": "schema absent"}, HTTPStatus.NOT_FOUND)
            if parts == ["runs"]:
                return self._json(api.runs(q))
            if len(parts) == 3 and parts[0] == "run":
                action = getattr(api, parts[2], None)
                if parts[2] in {"keys", "metrics", "status", "layers", "params", "log"} and action:
                    return self._json(action(parts[1], q))
            return self._json({"error": "route inconnue"}, HTTPStatus.NOT_FOUND)

        def _static(self, parts):
            target = (VIEWER / "/".join(parts)) if parts else VIEWER / "index.html"
            target = target.resolve()
            if VIEWER not in target.parents and target != VIEWER / "index.html":
                return self._send(b"interdit", "text/plain", HTTPStatus.FORBIDDEN)
            if not target.is_file():
                return self._send(b"introuvable", "text/plain", HTTPStatus.NOT_FOUND)
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            return self._send(target.read_bytes(), ctype)

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description="Visualiseur d'architecture et de runs MLflow")
    p.add_argument("--root", type=Path, default=HERE.parent,
                   help="racine du projet (defaut : parent de visualisation/)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    root = args.root.resolve()
    api = Api(root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(api))
    url = f"http://{args.host}:{args.port}/"
    n_arch = len(list(ARCH_DIR.glob("*.json")))
    print(f"[viz] projet   : {root}", flush=True)
    print(f"[viz] schemas  : {n_arch} dans {ARCH_DIR}", flush=True)
    state = "present" if api.store.root.is_dir() else "absent"
    print(f"[viz] mlruns   : {state} ({api.store.root})", flush=True)
    print(f"[viz] ouvert sur {url}  (Ctrl+C pour arreter)", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[viz] arrete")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
