"""Protein embeddings of every gene with ESM-2, for the hybrid model.

Reads the clean dataset (``tools/build_clean_dataset.py``) and writes, under
``benchmark/data/bacteria/clean/<mode>/``::

    esm2-<size>_genes.parquet   gene_idx, genome_id, feature_id, protein_idx
                                (-1 when the gene has no protein)
    esm2-<size>_full.npy        float16 (n_proteins, d), one row per distinct
                                protein
    esm2-<size>_pca<k>.npy      float16 (n_proteins, k), the PCA of the above,
                                the file shipped next to submission.py

How each embedding is made:

- protein = the RefSeq translation of the gene, or its CDS translated when the
  GenBank file has no ``/translation`` (64,908 genes of the train release);
- identical proteins are embedded once (484,222 distinct ones for 853,339
  genes);
- a protein longer than 1022 residues (the ESM-2 context) is cut in windows
  that overlap by half, averaged with a weight per residue;
- embedding = mean of the last layer over the residues, BOS and EOS left out;
- windows are sorted by length and batched under a token budget, in float16.

    python tools/esm_embeddings.py --mode train --model facebook/esm2_t33_650M_UR50D --pca 128
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_logger import RunLogger  # noqa: E402

DATA = ROOT / "benchmark" / "data" / "bacteria" / "clean"
MAX_RESIDUES = 1022
STRIDE = 511


def gene_proteins(seqs, genes, genomes, log):
    """Protein sequence of each gene, translated from the CDS if missing."""
    proteins = seqs["protein_seq"].to_numpy(dtype=object)
    table_of_genome = genomes.set_index("genome_idx")["transl_table"]
    tables = genes["genome_idx"].map(table_of_genome).fillna(11)\
        .astype(int).to_numpy()
    missing = np.flatnonzero(pd.isna(proteins))
    translated = 0
    if len(missing):
        from Bio.Seq import translate
    for i in missing:
        cds = seqs["cds_seq"].iat[i]
        if not isinstance(cds, str) or len(cds) < 6:
            continue
        try:
            protein = translate(cds, table=int(tables[i]), cds=True)
        except Exception:
            protein = translate(cds[:len(cds) // 3 * 3], table=int(tables[i]),
                                to_stop=True)
        if protein:
            proteins[i] = str(protein)
            translated += 1
    log(f"{len(missing):,} genes without a RefSeq protein, {translated:,} "
        f"translated from their CDS")
    return proteins


def windows(lengths):
    """(protein, start, end) of every window, long proteins cut in pieces."""
    items = []
    for p, n in enumerate(lengths):
        if n <= MAX_RESIDUES:
            items.append((p, 0, n))
            continue
        starts = list(range(0, n - MAX_RESIDUES, STRIDE)) + [n - MAX_RESIDUES]
        items.extend((p, s, s + MAX_RESIDUES) for s in starts)
    return np.array(items, dtype=np.int64)


class Encoder:
    """Token ids of protein strings through a lookup table.

    The ESM tokenizer of transformers is pure Python (0.7 s for 200
    proteins, half an hour for the whole release). Its vocabulary is one
    token per residue letter, so a 256-entry table gives the same ids.
    """

    def __init__(self, tokenizer):
        vocab = tokenizer.get_vocab()
        self.cls, self.eos = tokenizer.cls_token_id, tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id
        self.table = np.full(256, tokenizer.unk_token_id, dtype=np.int64)
        for token, idx in vocab.items():
            if len(token) == 1:
                self.table[ord(token)] = idx

    def __call__(self, texts):
        width = max(len(t) for t in texts) + 2
        ids = np.full((len(texts), width), self.pad, dtype=np.int64)
        for i, text in enumerate(texts):
            codes = self.table[np.frombuffer(text.encode("ascii", "replace"),
                                             dtype=np.uint8)]
            ids[i, 0] = self.cls
            ids[i, 1:1 + len(codes)] = codes
            ids[i, 1 + len(codes)] = self.eos
        ids = torch.as_tensor(ids)
        return ids, (ids != self.pad).long()


def pick_dtype(name, device):
    """bfloat16 on recent GPUs, float32 elsewhere.

    On Turing cards (GTX 16xx, T4) PyTorch has no fast float16 attention
    kernel, it materialises the attention matrices and spills.
    """
    if name != "auto":
        return getattr(torch, name)
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float32


def batches(items, budget):
    """Groups of windows of similar length, under ``budget`` tokens each."""
    order = np.argsort(-(items[:, 2] - items[:, 1]), kind="stable")
    out, current, width = [], [], 0
    for i in order:
        size = int(items[i, 2] - items[i, 1]) + 2       # BOS and EOS
        width = max(width, size)
        if current and width * (len(current) + 1) > budget:
            out.append(current)
            current, width = [], size
        current.append(i)
    if current:
        out.append(current)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", default="train", choices=["debug", "train"])
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--pca", type=int, default=128,
                        help="dimensions kept by the PCA")
    parser.add_argument("--tokens", type=int, default=32768,
                        help="token budget of a batch")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"])
    args = parser.parse_args()

    from transformers import AutoTokenizer, EsmModel

    root = DATA / args.mode
    size = args.model.split("_")[2].lower()           # 650m, 150m, 35m...
    tag = f"esm2-{size}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(args.dtype, device)

    with RunLogger(tag, mlflow=False) as run:
        run.config(vars(args))
        run.section("proteines")
        seqs = pq.read_table(root / "gene_sequences.parquet").to_pandas()
        genes = pd.read_parquet(root / "genes.parquet",
                                columns=["gene_idx", "genome_idx", "genome_id",
                                         "feature_id"])
        genomes = pd.read_parquet(root / "genomes.parquet",
                                  columns=["genome_idx", "transl_table"])
        proteins = gene_proteins(seqs, genes, genomes, run.log)
        seqs = None
        has = ~pd.isna(proteins)
        protein_idx = np.full(len(proteins), -1, dtype=np.int32)
        codes, unique = pd.factorize(proteins[has])
        protein_idx[has] = codes
        unique = [s.rstrip("*").replace("*", "X") for s in unique]
        lengths = np.array([len(s) for s in unique])
        items = windows(lengths)
        groups = batches(items, args.tokens)
        run.log(f"{len(unique):,} distinct proteins for {int(has.sum()):,} "
                f"genes, {int(lengths.sum()):,} residues, {len(items):,} "
                f"windows in {len(groups):,} batches")

        run.section("esm-2")
        encode = Encoder(AutoTokenizer.from_pretrained(args.model))
        model = EsmModel.from_pretrained(
            args.model, add_pooling_layer=False, torch_dtype=dtype,
            attn_implementation="sdpa").to(device).eval()
        dim = model.config.hidden_size
        gpu = torch.cuda.get_device_name() if device == "cuda" else "cpu"
        run.log(f"{args.model} on {gpu}, {dtype}, {dim} dimensions")
        total = np.zeros((len(unique), dim), dtype=np.float32)
        weight = np.zeros(len(unique), dtype=np.float32)
        t0, done = time.time(), 0
        with torch.inference_mode():
            for step, group in enumerate(run.track(groups, desc="esm2")):
                part = items[group]
                ids, mask = encode([unique[p][s:e] for p, s, e in part])
                ids, mask = ids.to(device), mask.to(device)
                hidden = model(input_ids=ids,
                               attention_mask=mask).last_hidden_state
                n_res = torch.as_tensor(part[:, 2] - part[:, 1], device=device)
                pos = torch.arange(ids.shape[1], device=device)[None, :]
                residue = (pos >= 1) & (pos <= n_res[:, None])
                mean = ((hidden.float() * residue[..., None]).sum(1)
                        / n_res[:, None]).cpu().numpy()
                w = (part[:, 2] - part[:, 1]).astype(np.float32)
                np.add.at(total, part[:, 0], mean * w[:, None])
                np.add.at(weight, part[:, 0], w)
                done += int(w.sum())
                if step % 50 == 0:
                    run.metric(step=step, **{
                        "residues_per_s": done / max(time.time() - t0, 1e-6)})
        emb = total / np.maximum(weight, 1.0)[:, None]
        total = None
        run.log(f"embeddings {emb.shape}, {time.time() - t0:.0f}s, "
                f"{done / max(time.time() - t0, 1e-6):,.0f} residues/s")

        run.section("sauvegarde")
        genes_out = genes[["gene_idx", "genome_id", "feature_id"]].copy()
        genes_out["protein_idx"] = protein_idx
        genes_out.to_parquet(root / f"{tag}_genes.parquet", index=False)
        np.save(root / f"{tag}_full.npy", emb.astype(np.float16))
        x = torch.as_tensor(emb, device=device)
        x = x - x.mean(0, keepdim=True)
        k = min(args.pca, dim)
        _, s, v = torch.pca_lowrank(x, q=min(k + 16, dim), center=False,
                                    niter=6)
        reduced = (x @ v[:, :k]).cpu().numpy().astype(np.float16)
        explained = float((s[:k] ** 2).sum() / (x ** 2).sum())
        np.save(root / f"{tag}_pca{k}.npy", reduced)
        run.log(f"PCA {k} dimensions, {explained:.1%} of the variance, "
                f"{reduced.nbytes / 1e6:.0f} MB")
        for name in (f"{tag}_genes.parquet", f"{tag}_full.npy",
                     f"{tag}_pca{k}.npy"):
            run.log(f"ecrit : {root / name}")
        run.done()


if __name__ == "__main__":
    main()
