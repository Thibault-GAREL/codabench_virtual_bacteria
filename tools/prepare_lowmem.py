"""Low-memory stand-in for ``benchopt prepare`` on a 16 GB machine.

``Dataset.prepare`` of ``benchmark/datasets/bacterial_expression.py`` holds
several full-size copies of the ~7.4e7 observations of the ``train`` release
(the ``pandas.merge`` MultiIndex, object arrays for the split labels,
``np.isin`` sorts, the boolean-mask copy written to parquet). On a laptop with
16 GB of RAM it fails with an ``_ArrayMemoryError``.

This script follows the same steps and writes the same files, with lean
arrays instead:

- ``attach`` looks the keys up with numpy on the category codes;
- the split labels are int8 codes, not Python strings;
- "seen in training" is a boolean lookup table on the pair codes;
- each region is written to parquet chunk by chunk.

``benchopt run`` only reads the prepared files (``get_data``), so the files
written here are used as is::

    python tools/prepare_lowmem.py --mode train

``--out`` writes the prepared layout elsewhere, to compare it with a layout
written by ``benchopt prepare`` (the raw files are still read from the usual
folder), and ``--chunk`` sets the rows written at a time.
"""

import argparse
import importlib.util
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark"
NAMES = ["train", "A", "B", "C", "D"]  # int8 code = position in this list


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def load_dataset_module():
    """Import ``bacterial_expression.py`` the way benchopt does, by path."""
    sys.path.insert(0, str(BENCHMARK))
    path = BENCHMARK / "datasets" / "bacterial_expression.py"
    spec = importlib.util.spec_from_file_location("bacterial_expression",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attach(obs, table, keys, cols, pair_codes):
    """Left-join ``cols`` of ``table`` onto ``obs`` by category codes.

    Same contract as the dataset's ``attach``: keys cast to the observation
    dtypes, keys absent from ``obs`` dropped, a duplicated key is an error, a
    missing key gives NaN, the row order of ``obs`` is kept.
    """
    table = table[keys + cols].astype({k: obs[k].dtype for k in keys})
    table = table.dropna(subset=keys)
    n_second = len(obs[keys[1]].cat.categories) + 1
    table_code = (
        table[keys[0]].cat.codes.to_numpy().astype(np.int64) * n_second
        + table[keys[1]].cat.codes.to_numpy().astype(np.int64))
    if len(np.unique(table_code)) != len(table_code):
        raise ValueError(f"duplicated keys {keys} in the joined table")

    order = np.argsort(table_code)
    sorted_code = table_code[order]
    obs_code = pair_codes(obs, keys)
    pos = np.searchsorted(sorted_code, obs_code)
    pos = np.minimum(pos, len(sorted_code) - 1)
    found = sorted_code[pos] == obs_code
    rows = order[pos]
    del obs_code, pos, order, sorted_code

    for col in cols:
        values = table[col].to_numpy(dtype=np.float64)[rows]
        values[~found] = np.nan
        obs[col] = values
    log(f"attached {cols} on {keys}, "
        f"{100 * (~found).mean():.2f}% of rows without a match")
    return obs


def seen_in(codes, fit):
    """``np.isin(codes, codes[fit])`` through a boolean lookup table."""
    table = np.zeros(int(codes.max()) + 1, dtype=bool)
    table[codes[fit]] = True
    return table[codes]


def write_part(obs, mask, path, chunk):
    """Write ``obs[mask].reset_index(drop=True)`` to parquet, by chunks."""
    writer = None
    n_rows = 0
    for start in range(0, len(obs), chunk):
        piece = obs.iloc[start:start + chunk][mask[start:start + chunk]]
        if len(piece) == 0 and writer is not None:
            continue
        table = pa.Table.from_pandas(piece, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
        n_rows += len(piece)
    writer.close()
    log(f"wrote {path.name}: {n_rows:,} rows")


def prepare(module, mode, chunk):
    """``Dataset.prepare`` of ``bacterial_expression.py``, step by step."""
    get_data_path = module.get_data_path
    raw = get_data_path("bacteria_raw")
    suffix = "_debug" if mode == "debug" else ""
    feat_name = f"feature_annotation{suffix}.csv"
    cond_name = f"condition_annotation{suffix}.csv"
    counts = f"read_count_{mode}.parquet"
    module.download_zenodo(module.PUBLIC_RECORD, [feat_name, cond_name], raw,
                           module.PUBLIC_TOKEN)
    if mode == "full":
        module.download_zenodo(module.PRIVATE_RECORD, [counts], raw,
                               os.environ.get(module.TOKEN_ENV))
    else:
        module.download_zenodo(module.PUBLIC_RECORD, [counts], raw,
                               module.PUBLIC_TOKEN)

    feat_df = pd.read_csv(raw / feat_name)
    cond_df = pd.read_csv(raw / cond_name)
    keys = module.GENE_KEYS + module.RUN_KEYS
    obs = pd.read_parquet(raw / counts).astype(
        {k: "category" for k in keys})
    log(f"read {len(obs):,} observations")
    obs = attach(obs, feat_df, module.GENE_KEYS, ["log_length_ratio"],
                 module.pair_codes)
    obs = attach(obs, cond_df, module.RUN_KEYS, ["log_size_factor"],
                 module.pair_codes)

    # Split label of each row as an int8 code, -1 when the status maps to
    # no part (the original maps it to NaN).
    held_out = "test" if mode == "full" else "eval"
    parts = {"Train": "train",
             **{f"{held_out}_{r}": r for r in module.REGIONS}}
    if mode == "full":
        parts.update({f"eval_{r}": "train" for r in module.REGIONS})
    status = obs.pop("status").astype("category")
    lookup = np.array([NAMES.index(parts[s]) if s in parts else -1
                       for s in status.cat.categories] + [-1], dtype=np.int8)
    part = lookup[status.cat.codes.to_numpy()]
    del status

    fit = part == 0
    new_gene = ~seen_in(module.pair_codes(obs, module.GENE_KEYS), fit)
    new_run = ~seen_in(module.pair_codes(obs, module.RUN_KEYS), fit)
    region = np.select([new_gene & new_run, new_gene, new_run],
                       [4, 3, 2], 1).astype(np.int8)
    del new_gene, new_run
    part = np.where(fit | (part == region), part, -1).astype(np.int8)
    del region, fit
    kept = {name: int((part == i).sum()) for i, name in enumerate(NAMES)}
    log(f"rows per part: {kept}, dropped: {int((part == -1).sum()):,}")

    root = get_data_path("bacteria") / mode
    root.mkdir(parents=True, exist_ok=True)
    genomes = root.parent / "genomes"
    zip_name = "genomes_debug.zip" if mode == "debug" else module.GENOMES
    if not all((genomes / f"{g}.gbff").exists()
               for g in feat_df["genome_id"].unique()):
        module.download_zenodo(module.PUBLIC_RECORD, [zip_name], raw,
                               module.PUBLIC_TOKEN)
        tmp = root.parent / "genomes.part"
        shutil.rmtree(tmp, ignore_errors=True)
        zipfile.ZipFile(raw / zip_name).extractall(tmp)
        genomes.mkdir(exist_ok=True)
        for path in (tmp / "genomes").iterdir():
            path.replace(genomes / path.name)
        shutil.rmtree(tmp)
        log(f"extracted the genomes to {genomes}")
    feat_df.to_csv(root / "feature_annotation.csv", index=False)
    cond_df.to_csv(root / "condition_annotation.csv", index=False)
    for i, name in enumerate(NAMES):
        write_part(obs, part == i, root / f"{name}.parquet", chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", default="train",
                        choices=["debug", "train", "full"])
    parser.add_argument("--out", type=Path, default=None,
                        help="folder for the prepared layout (default: the "
                        "benchmark data folder)")
    parser.add_argument("--chunk", type=int, default=5_000_000,
                        help="rows written to parquet at a time")
    args = parser.parse_args()

    module = load_dataset_module()
    data_home = Path(os.environ.get("BENCHOPT_DATA_HOME",
                                    BENCHMARK / "data"))
    out = args.out or data_home

    def get_data_path(key=None):
        base = data_home if key == "bacteria_raw" else out
        return (base / key).resolve() if key else base.resolve()

    module.get_data_path = get_data_path
    log(f"prepare BacterialExpression[mode={args.mode}] -> "
        f"{get_data_path('bacteria') / args.mode}")
    prepare(module, args.mode, args.chunk)
    log("done")


if __name__ == "__main__":
    main()
