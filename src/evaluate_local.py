"""Fit a submission on the local data and score it on the eval regions.

Does what the platform does, without the benchopt runner: loads the prepared
data with the dataset's own ``get_data``, builds the ``Solver`` of
``--submission`` through the same contract (``set_objective`` then ``run``),
then scores every region with the challenge metric. Everything goes to MLflow
(``mlruns/``) and to ``outputs/logs/<name>_run-<NN>/run.log`` through
RunLogger.

    python src/evaluate_local.py --submission submissions/glm_nb/submission.py --mode train
    python src/evaluate_local.py --submission ... --param steps=50 --param interaction=false

``load_data`` and ``evaluate`` are reused by ``tools/ablate.py``, which loads
the data once and fits many variants of one submission.
"""

import argparse
import gc
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark"
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_utils.nb_loss import (check_predictions, log_mae,  # noqa: E402
                                     log_r2, nb_nll)
from layer_stats import LayerStats  # noqa: E402
from run_logger import RunLogger  # noqa: E402

REGIONS = ["A", "B", "C", "D"]
CHALLENGE = ["A", "B", "C"]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # registered so that inspect.getfile finds the submission folder
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_value(text):
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def lean_read_parquet(path, *args, **kwargs):
    """``pd.read_parquet`` that hands Arrow's memory back as it converts.

    Same DataFrame as pandas returns, without keeping a second copy alive in
    the Arrow memory pool (7.5 GB of RAM for the 69 M training rows).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=kwargs.get("columns"))
    df = table.to_pandas(self_destruct=True, split_blocks=True)
    table = None
    pa.default_memory_pool().release_unused()
    return df


def memory_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return float("nan")


def load_data(mode, float32=True, log=print):
    """The dataset's own ``get_data``, with lean parquet reading.

    Returns the dict of ``get_data`` (``train_df``, ``samples_df``,
    ``genomes_dir``, ``test_sets`` and the meta tables).
    """
    dataset = load_module(BENCHMARK / "datasets" / "bacterial_expression.py",
                          "bacterial_expression")
    data_home = BENCHMARK / "data"
    dataset.get_data_path = (lambda key=None: (data_home / key).resolve()
                             if key else data_home.resolve())
    t = time.time()
    dataset.pd.read_parquet, original = lean_read_parquet, \
        dataset.pd.read_parquet
    try:
        data = dataset.Dataset.get_instance(mode=mode).get_data()
    finally:
        dataset.pd.read_parquet = original
    if float32:
        # Halves the float columns of the 69 M training rows. The only
        # difference with the platform, where the RAM is 32 GB.
        train_df = data["train_df"]
        for column in ("readcount", "log_length_ratio", "log_size_factor"):
            train_df[column] = train_df[column].astype(np.float32)
    log(f"train {len(data['train_df']):,} lignes | " + ", ".join(
        f"{k} {len(v[1]):,}" for k, v in data["test_sets"].items())
        + f" | {time.time() - t:.0f}s, RAM {memory_gb():.1f} Go")
    return data


def evaluate(module, submission, params, name, data, layer_every=10,
             extra_config=None, free_train=False):
    """Fit the submission's Solver with ``params``, score every region.

    One MLflow run and one run.log. Returns the dict of scores. With
    ``free_train`` the training rows are dropped after the fit, as the
    single run CLI does to keep the RAM low.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    test_sets = data["test_sets"]
    meta = {k: v for k, v in data.items()
            if k not in ("train_df", "samples_df", "genomes_dir", "test_sets")}
    with RunLogger(name) as run:
        run.config({"submission": str(Path(submission).relative_to(ROOT)),
                    "layer_stats_interval": layer_every,
                    **(extra_config or {}), **params})
        run.section("donnees")
        run.log(f"train {len(data['train_df']):,} lignes | " + ", ".join(
            f"{k} {len(v[1]):,}" for k, v in test_sets.items())
            + f" | RAM {memory_gb():.1f} Go")
        solver = module.Solver.get_instance(**params)
        solver.set_objective(train_df=data["train_df"],
                             samples_df=data["samples_df"],
                             genomes_dir=data["genomes_dir"], **meta)
        model = solver.model
        model.log, model.metric, model.track = run.log, run.metric, run.track
        if hasattr(model, "layer_stats") and layer_every > 0:
            model.layer_stats = (lambda module_: LayerStats(
                module_, every=layer_every))

        run.section("entrainement")
        t = time.time()
        solver.run(None)
        fit_s = time.time() - t
        model = solver.get_result()["model"]
        solver.train_df = None
        if free_train:
            data["train_df"] = None
        gc.collect()
        run.log(f"fit {fit_s:.0f}s, RAM {memory_gb():.1f} Go")

        run.section("evaluation")
        scores = {}
        t = time.time()
        for region, (features, counts) in test_sets.items():
            preds = model.predict(features, data["samples_df"],
                                  data["genomes_dir"])
            mu, sigma = check_predictions(preds, len(counts))
            scores[f"eval/nb_nll_{region}"] = nb_nll(counts, mu, sigma)
            scores[f"eval/mae_{region}"] = log_mae(counts, mu)
            scores[f"eval/r2_{region}"] = log_r2(counts, mu)
        predict_s = time.time() - t
        for metric in ("nb_nll", "mae", "r2"):
            keys = [f"eval/{metric}_{r}" for r in CHALLENGE
                    if f"eval/{metric}_{r}" in scores]
            if keys:
                scores[f"eval/{metric}_challenge"] = float(
                    np.mean([scores[k] for k in keys]))
        gpu = (torch.cuda.max_memory_allocated() / 1e9
               if torch is not None and torch.cuda.is_available()
               else float("nan"))
        scores.update({"time/fit_s": fit_s, "time/predict_s": predict_s,
                       "memory/gpu_peak_gb": gpu})
        run.metric(**scores)

        run.log("region   NB-NLL     MAE      R2   lignes")
        for region in REGIONS + ["challenge"]:
            if f"eval/nb_nll_{region}" not in scores:
                continue
            rows = len(test_sets[region][1]) if region in test_sets else ""
            run.log(f"{region:<9}{scores[f'eval/nb_nll_{region}']:>7.3f}"
                    f"{scores[f'eval/mae_{region}']:>8.3f}"
                    f"{scores[f'eval/r2_{region}']:>8.3f}   {rows}")
        run.log(f"fit {fit_s:.0f}s, predict {predict_s:.0f}s, "
                f"GPU pic {gpu:.2f} Go")
        run.done()
    del solver, model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--submission", required=True,
                        help="path of a submission.py")
    parser.add_argument("--mode", default="debug", choices=["debug", "train"])
    parser.add_argument("--param", action="append", default=[],
                        help="solver parameter, name=value (repeatable)")
    parser.add_argument("--name", default=None,
                        help="model name in the logs (default: the folder)")
    parser.add_argument("--layer-every", type=int, default=10,
                        help="log layer/... measures every N steps for the "
                        "visualiser (0 to turn off), if the model supports it")
    parser.add_argument("--float32", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="cast the float columns of train_df to float32 "
                        "(RAM), --no-float32 keeps the platform dtypes")
    args = parser.parse_args()

    submission = Path(args.submission).resolve()
    params = {k: parse_value(v) for k, v in
              (p.split("=", 1) for p in args.param)}
    name = args.name or submission.parent.name.replace("_", "-")
    module = load_module(submission, "submission")
    data = load_data(args.mode, args.float32)
    evaluate(module, submission, params, name, data, args.layer_every,
             extra_config={"mode": args.mode, "float32": args.float32},
             free_train=True)


if __name__ == "__main__":
    main()
