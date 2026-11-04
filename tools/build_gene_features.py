"""Per-gene feature file shipped next to the hybrid submission.

Combines the gene signals of the clean dataset and the ESM-2 protein
embeddings into one compressed file, keyed by (genome_id, feature_id)::

    gene_features.npz
        genome_id, feature_id   the gene keys (bytes)
        numeric       float16 (n_genes, F), standardised gene signals
        numeric_names
        protein_idx   int32 (n_genes,), row of ``esm``, -1 without protein
        esm           float16 (n_proteins, k), the ESM-2 PCA, scaled to a
                      unit mean variance

No read count enters it, so it is computed once offline for every gene, the
test genes included. The ESM rows are stored once per distinct protein.

    python tools/build_gene_features.py --mode train --esm esm2-650m
    python tools/build_gene_features.py --mode train --esm esm2-650m --pca 128 --out <folder>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmark" / "data" / "bacteria" / "clean"


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def gene_signals(genes):
    """Numeric gene signals, NaN filled, with a flag where it matters."""
    g = genes
    f = {
        "log_length": np.log(g["length"].clip(lower=1)),
        "log_length_ratio": g["log_length_ratio"],
        "cai": g["cai"], "cai_known": g["cai"].notna(),
        "gc_cds": g["gc_cds"], "gc3": g["gc3"], "gc_upstream": g["gc_upstream"],
        "sd_score": g["sd_score"], "sd_spacing": g["sd_spacing"],
        "sigma70_score": g["sigma70_score"], "sigma70_m10": g["sigma70_m10"],
        "sigma70_m35": g["sigma70_m35"], "sigma70_dist": g["sigma70_dist"],
        "log_operon_rank": np.log(g["operon_rank"].clip(lower=1)),
        "log_operon_size": np.log(g["operon_size"].clip(lower=1)),
        "operon_first": g["operon_rank"] == 1,
        "up_dist": signed_log(g["up_dist"]), "up_known": g["up_dist"].notna(),
        "down_dist": signed_log(g["down_dist"]),
        "down_known": g["down_dist"].notna(),
        "up_same_strand": g["up_same_strand"].astype(float),
        "down_same_strand": g["down_same_strand"].astype(float),
        "ori_position": g["ori_position"],
        "ori_known": g["ori_position"].notna(),
        "leading_strand": g["leading_strand"].astype(float),
        "on_chromosome": g["replicon_type"] == "chromosome",
        "on_plasmid": g["replicon_type"] == "plasmid",
        "on_contig": g["replicon_type"] == "contig",
        "ribosomal": g["is_ribosomal_protein"],
        "hypothetical": g["is_hypothetical"], "partial": g["is_partial"],
        "split": g["is_split"], "pseudo": g["is_pseudo"],
        "reference": g["is_reference"], "homology": g["homology"],
        "log_copies": np.log(g["n_copies"].clip(lower=1)),
        "start_atg": g["start_codon"] == "ATG",
        "start_gtg": g["start_codon"] == "GTG",
        "start_ttg": g["start_codon"] == "TTG",
    }
    table = pd.DataFrame({k: pd.to_numeric(pd.Series(v), errors="coerce")
                          .astype(float) for k, v in f.items()})
    # a missing value becomes the column mean, i.e. 0 once standardised
    mean, std = table.mean(), table.std().replace(0, 1).fillna(1)
    table = ((table - mean) / std).fillna(0.0)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", default="train", choices=["debug", "train"])
    parser.add_argument("--esm", default="esm2-650m",
                        help="prefix of the files written by esm_embeddings")
    parser.add_argument("--pca", type=int, default=128,
                        help="PCA dimensions of the ESM file to read")
    parser.add_argument("--out", type=Path, default=None,
                        help="folder of gene_features.npz (default: the "
                        "clean data folder)")
    args = parser.parse_args()

    root = DATA / args.mode
    genes = pd.read_parquet(root / "genes.parquet")
    signals = gene_signals(genes)

    mapping = pd.read_parquet(root / f"{args.esm}_genes.parquet")
    assert np.array_equal(mapping["gene_idx"].to_numpy(),
                          genes["gene_idx"].to_numpy())
    esm = np.load(root / f"{args.esm}_pca{args.pca}.npy").astype(np.float32)
    esm /= np.sqrt((esm ** 2).mean(axis=0).mean())

    out = (args.out or root) / "gene_features.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        genome_id=genes["genome_id"].astype(str).to_numpy().astype("S"),
        feature_id=genes["feature_id"].astype(str).to_numpy().astype("S"),
        numeric=signals.to_numpy().astype(np.float16),
        numeric_names=np.array(signals.columns, dtype="S"),
        protein_idx=mapping["protein_idx"].to_numpy().astype(np.int32),
        esm=esm.astype(np.float16))
    print(f"{out}: {len(genes):,} genes, {signals.shape[1]} signals, "
          f"ESM {esm.shape}, {out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
