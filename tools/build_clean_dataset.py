"""Build a clean, model-ready version of the prepared bacterial dataset.

Reads a prepared layout (``benchmark/data/bacteria/<mode>``, written by
``benchopt prepare`` or ``tools/prepare_lowmem.py``) and the GenBank genomes,
then writes under ``benchmark/data/bacteria/clean/<mode>/``::

    genomes.parquet         one row per genome (``genome_idx``)
    genes.parquet           one row per gene, ``gene_idx`` = row of
                            feature_annotation.csv
    gene_sequences.parquet  oriented upstream, gene and protein sequences
    runs.parquet            one row per run, ``run_idx`` = row of
                            condition_annotation.csv
    obs_<split>.parquet     gene_idx, run_idx, readcount, offset, log_expr
    report.json             counts and the checks run at the end

Every sequence reads in the gene's own direction (5' to 3'), so the strand
sign is gone. Raw coordinates are replaced by what they mean:

- ``replicon_type`` chromosome, plasmid or contig of a draft assembly, and
  ``replicon_length``;
- ``up_*`` and ``down_*``: the neighbour gene before and after, in reading
  order, its distance (negative = overlap) and whether it is on the same
  strand (tandem) or not (divergent upstream, convergent downstream);
- ``operon_id``, ``operon_rank`` (1 = first gene, the one that carries the
  promoter), ``operon_size``: runs of same-strand genes closer than
  ``--operon-gap`` bp;
- ``ori_position`` (0 at the replication origin, 1 at the terminus) and
  ``leading_strand``, from the cumulative GC skew of each chromosome of a
  complete assembly (NaN on draft assemblies and unclear skews);
- ``is_partial`` (gene cut by the end of a contig) and ``is_split`` (gene in
  several pieces).

The raw ``Chr``, ``Start``, ``End`` and ``Strand`` stay in
feature_annotation.csv, at the same row as ``gene_idx``.

    python tools/build_clean_dataset.py --mode train --workers 6
"""

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmark" / "data" / "bacteria"
SPLITS = ["train", "A", "B", "C", "D"]
# A chromosome shorter than this is not given a replication origin.
MIN_CHROMOSOME = 500_000
# Smallest arm / half chromosome accepted for a GC skew origin (1 = equal arms).
MIN_ARM_BALANCE = 0.7
# "Picard" is a collection of natural E. coli isolates, not a species.
SPECIES_FIX = {"Picard": "Escherichia_coli"}
FEATURE_PRIORITY = {"CDS": 0, "rRNA": 1, "tRNA": 1, "ncRNA": 1, "tmRNA": 1,
                    "misc_RNA": 2, "gene": 3}
SEQ_SCHEMA = pa.schema([("gene_idx", pa.int32()),
                        ("upstream_seq", pa.string()),
                        ("cds_seq", pa.string()),
                        ("protein_seq", pa.string())])


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


# --------------------------------------------------------------------------
# Genome side, one worker per genome
# --------------------------------------------------------------------------

def record_kind(record):
    """Chromosome, plasmid or contig (piece of a draft assembly)."""
    desc = record.description.lower()
    source = next((f for f in record.features if f.type == "source"), None)
    if "plasmid" in desc or (source is not None
                             and "plasmid" in source.qualifiers):
        return "plasmid"
    keywords = [k.lower() for k in record.annotations.get("keywords", [])]
    draft_words = ("whole genome shotgun", "contig", "scaffold")
    if "wgs" in keywords or any(w in desc for w in draft_words):
        return "contig"
    return "chromosome"


def gc_skew_origin(seq):
    """Origin and terminus from the cumulative GC skew (0-based positions).

    On the leading strand G outnumbers C, so the cumulative G - C count
    falls to its minimum at the origin and peaks at the terminus.
    """
    a = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
    g = ((a == 71) | (a == 103)).astype(np.int32)
    c = ((a == 67) | (a == 99)).astype(np.int32)
    cum = np.cumsum(g - c)
    return int(np.argmin(cum)), int(np.argmax(cum))


def replication_arms(x, strand, ori, ter, length, circular):
    """Position between origin (0) and terminus (1), and leading strand."""
    if circular:
        right = max((ter - ori) % length, 1)
        clockwise = (x - ori) % length
        on_right = clockwise <= right
        pos = np.where(on_right, clockwise / right,
                       (length - clockwise) / max(length - right, 1))
    else:
        on_right = x >= ori
        pos = np.where(on_right, (x - ori) / max(length - ori, 1),
                       (ori - x) / max(ori, 1))
    leading = np.where(on_right, strand == "+", strand == "-")
    return np.clip(pos, 0.0, 1.0), leading


def arm_balance(ori, ter, length, circular):
    """1 when both replication arms have the same length."""
    if circular:
        right = (ter - ori) % length
        return min(right, length - right) / (length / 2)
    return min(ori, length - ori) / (length / 2)


def parse_parts(start, end):
    starts = [int(v) for v in str(start).split(";")]
    ends = [int(v) for v in str(end).split(";")]
    return tuple(sorted(zip(starts, ends)))


def gc_fraction(seq):
    if not seq:
        return np.nan
    s = seq.upper()
    return (s.count("G") + s.count("C")) / len(s)


def lineage_ranks(lineage):
    """Phylum, order, family and genus from a GenBank lineage."""
    def pick(suffix):
        return next((t for t in lineage if t.endswith(suffix)), None)
    family = pick("aceae")
    genus = None
    if family is not None and lineage.index(family) + 1 < len(lineage):
        genus = lineage[lineage.index(family) + 1]
    return {"domain": lineage[0] if lineage else None,
            "phylum": pick("ota"), "order": pick("ales"),
            "family": family, "genus": genus}


def process_genome(task):
    """Features and oriented sequences of the genes of one genome."""
    from Bio import SeqIO
    from Bio.Seq import reverse_complement, translate

    genome_id, path, genes, upstream, operon_gap = task
    records = list(SeqIO.parse(str(path), "genbank"))
    by_name = {}
    for rec in records:
        by_name[rec.id] = by_name[rec.name] = rec

    # Replicon kind, sequence and replication origin of every record
    info, index = {}, {}
    for rec in records:
        kind = record_kind(rec)
        seq = str(rec.seq)
        circular = rec.annotations.get("topology") == "circular"
        ori = ter = balance = None
        if kind == "chromosome" and len(seq) >= MIN_CHROMOSOME:
            ori, ter = gc_skew_origin(seq)
            balance = arm_balance(ori, ter, len(seq), circular)
        info[id(rec)] = {"kind": kind, "seq": seq, "length": len(seq),
                         "circular": circular, "ori": ori, "ter": ter,
                         "balance": balance}
        for feat in rec.features:
            prio = FEATURE_PRIORITY.get(feat.type)
            if prio is None:
                continue
            parts = tuple(sorted((int(p.start) + 1, int(p.end))
                                 for p in feat.location.parts))
            strand = "-" if feat.location.strand == -1 else "+"
            key = (id(rec), strand, parts)
            if key not in index or prio < FEATURE_PRIORITY[index[key].type]:
                index[key] = feat

    n = len(genes)
    out = {name: np.full(n, np.nan) for name in (
        "replicon_length", "ori_position", "up_dist", "down_dist",
        "gc_cds", "gc3", "gc_upstream")}
    out.update({name: np.full(n, -1, dtype=np.int64) for name in (
        "up_gene_idx", "down_gene_idx", "operon_local", "operon_rank",
        "operon_size")})
    replicon_type = np.full(n, "missing", dtype=object)
    circular = np.zeros(n, dtype=bool)
    leading = np.full(n, np.nan)
    up_same = np.full(n, np.nan)
    down_same = np.full(n, np.nan)
    text = {name: [None] * n for name in (
        "gene_name", "product", "locus_tag", "protein_id", "ec_number",
        "feature_type", "start_codon", "stop_codon")}
    flags = {name: np.zeros(n, dtype=bool) for name in (
        "matched_gbff", "is_pseudo", "translation_ok", "translation_checked",
        "length_ok")}
    seq_rows = {"upstream_seq": [None] * n, "cds_seq": [None] * n,
                "protein_seq": [None] * n}

    chrom = genes["Chr"].astype(str).str.split(";").str[0].to_numpy()
    strand = genes["Strand"].astype(str).str[0].to_numpy()
    parts = [parse_parts(s, e) for s, e in zip(genes["Start"], genes["End"])]
    lo = np.array([p[0][0] for p in parts])
    hi = np.array([max(e for _, e in p) for p in parts])
    lengths = genes["Length"].to_numpy()
    gene_idx = genes["gene_idx"].to_numpy()

    # Per gene: sequences, annotations, checks
    for i in range(n):
        rec = by_name.get(chrom[i])
        if rec is None:
            continue
        rinfo = info[id(rec)]
        seq = rinfo["seq"]
        replicon_type[i] = rinfo["kind"]
        out["replicon_length"][i] = rinfo["length"]
        circular[i] = rinfo["circular"]
        feat = index.get((id(rec), strand[i], parts[i]))
        if feat is not None:
            cds = str(feat.extract(rec.seq))
            q = feat.qualifiers
            flags["matched_gbff"][i] = True
            text["feature_type"][i] = feat.type
            for col, key in (("gene_name", "gene"), ("product", "product"),
                             ("locus_tag", "locus_tag"),
                             ("protein_id", "protein_id"),
                             ("ec_number", "EC_number")):
                text[col][i] = q.get(key, [None])[0]
            flags["is_pseudo"][i] = "pseudo" in q or "pseudogene" in q
            protein = q.get("translation", [None])[0]
            seq_rows["protein_seq"][i] = protein
            fuzzy = "<" in str(feat.location) or ">" in str(feat.location)
            if (protein and feat.type == "CDS" and not fuzzy
                    and not flags["is_pseudo"][i]):
                flags["translation_checked"][i] = True
                try:
                    table = int(q.get("transl_table", ["11"])[0])
                    flags["translation_ok"][i] = (
                        str(translate(cds, table=table, cds=True)) == protein)
                except Exception:
                    flags["translation_ok"][i] = False
        else:
            cds = "".join(seq[s - 1:e] for s, e in parts[i])
            if strand[i] == "-":
                cds = reverse_complement(cds)
        if strand[i] == "+":
            up = seq[max(lo[i] - 1 - upstream, 0):lo[i] - 1]
        else:
            up = reverse_complement(seq[hi[i]:hi[i] + upstream])
        seq_rows["cds_seq"][i] = cds
        seq_rows["upstream_seq"][i] = up
        flags["length_ok"][i] = len(cds) == lengths[i]
        out["gc_cds"][i] = gc_fraction(cds)
        out["gc3"][i] = gc_fraction(cds[2::3])
        out["gc_upstream"][i] = gc_fraction(up)
        text["start_codon"][i] = cds[:3].upper()
        text["stop_codon"][i] = cds[-3:].upper()

    # Per replicon: neighbours, operons, replication arms
    operon_base = 0
    for name in pd.unique(chrom):
        rec = by_name.get(name)
        if rec is None:
            continue
        rinfo = info[id(rec)]
        rows = np.flatnonzero(chrom == name)
        rows = rows[np.argsort(lo[rows], kind="stable")]
        s, l_, h = strand[rows], lo[rows], hi[rows]
        gap = np.full(len(rows), np.nan)  # gap[k] = space between k-1 and k
        gap[1:] = l_[1:] - h[:-1] - 1
        nxt_gap = np.append(gap[1:], np.nan)
        prev_row = np.append(-1, gene_idx[rows][:-1])
        next_row = np.append(gene_idx[rows][1:], -1)
        prev_same = np.append(np.nan, (s[1:] == s[:-1]).astype(float))
        next_same = np.append((s[1:] == s[:-1]).astype(float), np.nan)
        plus = s == "+"
        out["up_gene_idx"][rows] = np.where(plus, prev_row, next_row)
        out["down_gene_idx"][rows] = np.where(plus, next_row, prev_row)
        out["up_dist"][rows] = np.where(plus, gap, nxt_gap)
        out["down_dist"][rows] = np.where(plus, nxt_gap, gap)
        up_same[rows] = np.where(plus, prev_same, next_same)
        down_same[rows] = np.where(plus, next_same, prev_same)

        link = np.zeros(len(rows), dtype=bool)
        link[1:] = (s[1:] == s[:-1]) & (gap[1:] < operon_gap)
        group = np.cumsum(~link) - 1
        size = np.bincount(group)
        first = np.searchsorted(group, group)        # first row of the group
        rank_left = np.arange(len(rows)) - first + 1
        rank = np.where(plus, rank_left, size[group] - rank_left + 1)
        out["operon_local"][rows] = operon_base + group
        out["operon_rank"][rows] = rank
        out["operon_size"][rows] = size[group]
        operon_base += len(size)

        if rinfo["ori"] is not None and rinfo["balance"] >= MIN_ARM_BALANCE:
            mid = (l_ + h) / 2 - 1
            pos, lead = replication_arms(mid, s, rinfo["ori"], rinfo["ter"],
                                         rinfo["length"], rinfo["circular"])
            out["ori_position"][rows] = pos
            leading[rows] = lead.astype(float)

    features = pd.DataFrame({
        "gene_idx": gene_idx,
        "replicon_type": replicon_type,
        "replicon_length": out["replicon_length"],
        "replicon_circular": circular,
        "ori_position": out["ori_position"],
        "leading_strand": leading,
        "up_gene_idx": out["up_gene_idx"],
        "up_dist": out["up_dist"],
        "up_same_strand": up_same,
        "down_gene_idx": out["down_gene_idx"],
        "down_dist": out["down_dist"],
        "down_same_strand": down_same,
        "operon_local": out["operon_local"],
        "operon_rank": out["operon_rank"],
        "operon_size": out["operon_size"],
        "gc_cds": out["gc_cds"], "gc3": out["gc3"],
        "gc_upstream": out["gc_upstream"],
        **text, **flags,
    })
    sequences = pd.DataFrame({"gene_idx": gene_idx, **seq_rows})

    kinds = [info[id(r)]["kind"] for r in records]
    chroms = [info[id(r)] for r in records if info[id(r)]["ori"] is not None]
    main = max(chroms, key=lambda r: r["length"]) if chroms else None
    total = "".join(info[id(r)]["seq"] for r in records)
    lineage = records[0].annotations.get("taxonomy", []) if records else []
    genome = {
        "genome_id": genome_id,
        "n_records": len(records),
        "n_chromosomes": kinds.count("chromosome"),
        "n_plasmids": kinds.count("plasmid"),
        "n_contigs": kinds.count("contig"),
        "assembly_complete": "contig" not in kinds,
        "genome_length": len(total),
        "gc_genome": gc_fraction(total),
        "ori": main["ori"] if main else None,
        "ter": main["ter"] if main else None,
        "arm_balance": main["balance"] if main else None,
        "lineage": "; ".join(lineage),
        **lineage_ranks(lineage),
    }
    return genome, features, sequences


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

MEDIUM_WORDS = re.compile(r"\b(broth|medium|media|minimal|liquid|agar)\b")
MEDIUM_SYNONYMS = {
    "lysogeny": "lb", "luria-bertani": "lb", "luria bertani": "lb",
    "tryptic soy": "tsb", "trypticase soy": "tsb",
    "brain heart infusion": "bhi", "middlebrook 7h9": "7h9",
    "mueller-hinton": "mhb", "mueller hinton": "mhb", "mh": "mhb",
}
COMPOUND_SYNONYMS = {"h2o2": "hydrogen peroxide", "atc": "anhydrotetracycline",
                     "mem": "meropenem"}


def clean_name(value, generic=None, synonyms=None):
    """Lower case, no parenthesis, no generic word, one spelling."""
    if not isinstance(value, str):
        return None
    pieces = []
    for piece in value.lower().split("+"):
        s = re.sub(r"\([^)]*\)", " ", piece)
        bare = re.sub(r"\s+", " ", s).strip()
        if generic is not None:
            s = generic.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip() or bare
        s = (synonyms or {}).get(s, s)
        s = re.sub(r"(?<=\w)-(?=\w)", "", s)
        if s:
            pieces.append(s)
    return " + ".join(pieces) or None


def clean_runs(runs, genome_idx):
    out = pd.DataFrame({
        "run_idx": np.arange(len(runs), dtype=np.int32),
        "run_accession": runs["run_accession"].astype(str),
        "genome_id": runs["genome_id"].astype(str),
        "genome_idx": runs["genome_id"].map(genome_idx).astype(np.int32),
        "species": runs["species"].replace(SPECIES_FIX),
        "species_label": runs["species"],
        "prj_idx": pd.factorize(runs["prj_accession"])[0].astype(np.int32),
        "exp_idx": pd.factorize(runs["experiment_accession"])[0]
        .astype(np.int32),
        "log_size_factor": runs["log_size_factor"].astype(np.float32),
        "has_sheet": runs["condition_confidence"].notna(),
    })
    out["condition_confidence"] = pd.Categorical(
        runs["condition_confidence"].fillna("unknown"),
        categories=["unknown", "low", "medium", "high"], ordered=True)
    for col in ("is_wild_type", "has_perturbation"):
        values = runs[col].map({True: "true", False: "false",
                                "True": "true", "False": "false"})
        out[col] = pd.Categorical(values.fillna("unknown"),
                                  categories=["unknown", "false", "true"])
    for col in ("medium_category", "oxygen_condition", "growth_mode",
                "growth_phase", "perturbation_category"):
        out[col] = runs[col].fillna("unknown").astype("category")
    medium = runs["medium_name"].map(
        lambda v: clean_name(v, MEDIUM_WORDS, MEDIUM_SYNONYMS))
    out["medium_name"] = medium.fillna("unknown").astype("category")
    out["medium_base"] = (medium.str.split(" + ", regex=False).str[0]
                          .fillna("unknown").astype("category"))
    compound = runs["perturbation_compound"].map(
        lambda v: clean_name(v, None, COMPOUND_SYNONYMS))
    out["perturbation_compound"] = (compound.fillna("unknown")
                                    .astype("category"))
    for col in ("temperature_celsius", "ph", "agitation_rpm", "od600"):
        out[col] = pd.to_numeric(runs[col], errors="coerce").astype(
            np.float32)
    cond = ["is_wild_type", "medium_category", "medium_name",
            "temperature_celsius", "ph", "oxygen_condition", "agitation_rpm",
            "growth_mode", "growth_phase", "od600", "has_perturbation",
            "perturbation_category", "perturbation_compound"]
    out["n_known_conditions"] = runs[cond].notna().sum(axis=1).astype(
        np.int8)
    levels = {
        "medium_name": [int(runs["medium_name"].nunique()),
                        int(medium.nunique())],
        "perturbation_compound": [int(runs["perturbation_compound"]
                                      .nunique()), int(compound.nunique())],
    }
    return out, levels


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------

def lookup_gene_idx(obs, genes):
    """gene_idx of each observation, by category codes (no strings)."""
    gid = genes["genome_id"].astype(obs["genome_id"].dtype).cat.codes
    fid = genes["feature_id"].astype(obs["feature_id"].dtype).cat.codes
    width = len(obs["feature_id"].cat.categories) + 1
    keep = (gid.to_numpy() >= 0) & (fid.to_numpy() >= 0)
    table_code = (gid.to_numpy()[keep].astype(np.int64) * width
                  + fid.to_numpy()[keep])
    table_idx = genes["gene_idx"].to_numpy()[keep]
    order = np.argsort(table_code)
    sorted_code = table_code[order]
    code = (obs["genome_id"].cat.codes.to_numpy().astype(np.int64) * width
            + obs["feature_id"].cat.codes.to_numpy())
    pos = np.minimum(np.searchsorted(sorted_code, code), len(order) - 1)
    found = sorted_code[pos] == code
    idx = table_idx[order[pos]].astype(np.int32)
    idx[~found] = -1
    return idx


def lookup_run_idx(obs, runs):
    codes = runs["run_accession"].astype(obs["run_accession"].dtype)\
        .cat.codes.to_numpy()
    table = np.full(len(obs["run_accession"].cat.categories), -1, np.int32)
    table[codes[codes >= 0]] = runs["run_idx"].to_numpy()[codes >= 0]
    return table[obs["run_accession"].cat.codes.to_numpy()]


def build_obs(src, out, split, genes, runs, llr, lsf):
    """Write obs_<split>.parquet and check it against the source rows."""
    path = src / f"{split}.parquet"
    obs = pq.read_table(path, columns=["genome_id", "feature_id",
                                       "run_accession", "readcount"]
                        ).to_pandas()
    gene_idx = lookup_gene_idx(obs, genes)
    run_idx = lookup_run_idx(obs, runs)
    count = obs["readcount"].to_numpy()
    del obs
    check = {"rows": int(len(count)),
             "unmatched_genes": int((gene_idx < 0).sum()),
             "unmatched_runs": int((run_idx < 0).sum()),
             "non_integer_counts": int((count != np.round(count)).sum()),
             "max_count": float(count.max())}
    count = count.astype(np.int32)

    # The offsets gathered from the tables must equal the source columns
    src_llr = pq.read_table(path, columns=["log_length_ratio"])\
        .column(0).to_numpy()
    check["llr_equal"] = bool(np.array_equal(src_llr, llr[gene_idx]))
    del src_llr
    src_lsf = pq.read_table(path, columns=["log_size_factor"])\
        .column(0).to_numpy()
    check["lsf_equal"] = bool(np.array_equal(src_lsf, lsf[run_idx]))
    del src_lsf

    offset = (llr[gene_idx] + lsf[run_idx]).astype(np.float32)
    log_expr = (np.log(count + 0.5) - offset).astype(np.float32)
    table = pa.table({"gene_idx": gene_idx, "run_idx": run_idx,
                      "readcount": count, "offset": offset,
                      "log_expr": log_expr})
    pq.write_table(table, out / f"obs_{split}.parquet", compression="zstd",
                   row_group_size=4_000_000)
    log(f"obs_{split}: {check}")
    return check


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", default="train",
                        choices=["debug", "train", "full"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--upstream", type=int, default=200,
                        help="bp of upstream sequence kept per gene")
    # 100 bp keeps thrLABC and lacZYA (gaps 81, 51, 65) whole, and leaves
    # lacI (122 bp from lacZ) out. The exact gaps stay in up_dist, down_dist.
    parser.add_argument("--operon-gap", type=int, default=100,
                        help="max bp between two genes of one operon")
    args = parser.parse_args()

    src = DATA / args.mode
    out = DATA / "clean" / args.mode
    out.mkdir(parents=True, exist_ok=True)
    genomes_dir = DATA / "genomes"
    t0 = time.time()
    log(f"clean dataset {args.mode}: {src} -> {out}")

    genes = pd.read_csv(src / "feature_annotation.csv", low_memory=False)
    runs = pd.read_csv(src / "condition_annotation.csv", low_memory=False)
    genes["gene_idx"] = np.arange(len(genes), dtype=np.int32)
    genome_ids = sorted(genes["genome_id"].unique())
    genome_idx = {g: i for i, g in enumerate(genome_ids)}
    log(f"{len(genes):,} genes, {len(runs):,} runs, {len(genome_ids)} genomes")

    # 1. Genomes, in parallel
    cols = ["gene_idx", "Chr", "Start", "End", "Strand", "Length"]
    tasks = [(gid, genomes_dir / f"{gid}.gbff", sub[cols].reset_index(
        drop=True), args.upstream, args.operon_gap)
        for gid, sub in genes.groupby("genome_id", sort=True)]
    genome_rows, feature_parts = [], []
    writer = pq.ParquetWriter(out / "gene_sequences.parquet", SEQ_SCHEMA,
                              compression="zstd")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, (genome, feats, seqs) in enumerate(
                pool.map(process_genome, tasks, chunksize=1), 1):
            genome_rows.append(genome)
            feature_parts.append(feats)
            writer.write_table(pa.Table.from_pandas(
                seqs, schema=SEQ_SCHEMA, preserve_index=False))
            if k % 10 == 0 or k == len(tasks):
                elapsed = time.time() - t0
                eta = elapsed / k * (len(tasks) - k)
                log(f"genomes {k:>3}/{len(tasks)} ({100 * k / len(tasks):3.0f}%)"
                    f"  ETA {eta / 60:4.1f} min  last {genome['genome_id']}")
    writer.close()

    # 2. Genome table
    gtab = pd.DataFrame(genome_rows)
    sp = genes.drop_duplicates("genome_id").set_index("genome_id")["species"]
    gtab.insert(0, "genome_idx", gtab["genome_id"].map(genome_idx)
                .astype(np.int16))
    gtab.insert(2, "species", gtab["genome_id"].map(sp).replace(SPECIES_FIX))
    gtab.insert(3, "species_label", gtab["genome_id"].map(sp))
    gtab["n_genes"] = gtab["genome_id"].map(genes["genome_id"]
                                            .value_counts()).astype(np.int32)
    gtab["n_runs"] = gtab["genome_id"].map(runs["genome_id"].value_counts())\
        .fillna(0).astype(np.int32)
    gtab = gtab.sort_values("genome_idx").reset_index(drop=True)
    gtab.to_parquet(out / "genomes.parquet", index=False)

    # 3. Gene table
    feats = pd.concat(feature_parts, ignore_index=True)\
        .sort_values("gene_idx").reset_index(drop=True)
    assert np.array_equal(feats["gene_idx"].to_numpy(),
                          genes["gene_idx"].to_numpy())
    # operon ids unique over the whole dataset
    genome_of = genes["genome_id"].map(genome_idx).to_numpy()
    feats["operon_id"] = pd.factorize(
        pd.Series(genome_of).astype(str) + ":" + feats["operon_local"]
        .astype(str))[0].astype(np.int32)
    ortho_key = np.where(genes["ref_feature_id"].notna(),
                         genes["ref_genome_id"].astype(str) + "|"
                         + genes["ref_feature_id"].astype(str),
                         genes["genome_id"].astype(str) + "|"
                         + genes["feature_id"].astype(str))
    starts = genes["Start"].astype(str)
    clean = pd.DataFrame({
        "gene_idx": genes["gene_idx"],
        "genome_idx": genome_of.astype(np.int16),
        "genome_id": genes["genome_id"].astype("category"),
        "feature_id": genes["feature_id"].astype(str),
        "species": genes["species"].replace(SPECIES_FIX).astype("category"),
        "species_label": genes["species"].astype("category"),
        "ortho_group": pd.factorize(ortho_key)[0].astype(np.int32),
        "is_reference": genes["ref_feature_id"].isna(),
        "homology": genes["homology"].fillna(1.0).astype(np.float32),
        "protein_group": pd.factorize(genes["feature_id"])[0]
        .astype(np.int32),
        "length": genes["Length"].astype(np.int32),
        "log_length_ratio": genes["log_length_ratio"].astype(np.float32),
        "is_partial": genes["on_contig_edge"].astype(bool),
        "is_split": starts.str.contains(";", regex=False),
    })
    feats = feats.drop(columns=["gene_idx", "operon_local"])
    for col in ("replicon_length", "up_dist", "down_dist"):
        feats[col] = feats[col].astype(np.float32)
    for col in ("ori_position", "gc_cds", "gc3", "gc_upstream"):
        feats[col] = feats[col].astype(np.float32)
    for col in ("up_gene_idx", "down_gene_idx"):
        feats[col] = feats[col].astype(np.int32)
    for col in ("operon_rank", "operon_size"):
        feats[col] = feats[col].astype(np.int16)
    for col in ("leading_strand", "up_same_strand", "down_same_strand"):
        feats[col] = feats[col].map({1.0: True, 0.0: False}).astype("boolean")
    for col in ("replicon_type", "feature_type", "start_codon", "stop_codon"):
        feats[col] = feats[col].astype("category")
    gene_table = pd.concat([clean, feats], axis=1)
    gene_table.to_parquet(out / "genes.parquet", index=False)
    log(f"genes.parquet: {gene_table.shape}")

    # 4. Run table
    run_table, levels = clean_runs(runs, genome_idx)
    run_table.to_parquet(out / "runs.parquet", index=False)
    log(f"runs.parquet: {run_table.shape}, levels {levels}")

    # 5. Observations
    llr = genes["log_length_ratio"].to_numpy()
    lsf = runs["log_size_factor"].to_numpy()
    obs_checks = {}
    for split in SPLITS:
        if (src / f"{split}.parquet").exists():
            obs_checks[split] = build_obs(src, out, split, genes, run_table,
                                          llr, lsf)

    # 6. Report
    checked = gene_table["translation_checked"]
    strand = genes["Strand"].astype(str).str[0]
    report = {
        "mode": args.mode, "built": time.strftime("%Y-%m-%d %H:%M"),
        "minutes": round((time.time() - t0) / 60, 1),
        "params": vars(args),
        "genes": int(len(gene_table)), "runs": int(len(run_table)),
        "genomes": int(len(gtab)),
        "matched_gbff": float(gene_table["matched_gbff"].mean()),
        "translation_ok": {s: float(gene_table.loc[checked & (strand == s),
                                                   "translation_ok"].mean())
                           for s in ("+", "-")},
        "translation_checked": int(checked.sum()),
        "length_ok": float(gene_table["length_ok"].mean()),
        "start_codons": {k: int(v) for k, v in gene_table["start_codon"]
                         .value_counts().head(5).items()},
        "replicon_type": {k: int(v) for k, v in gene_table["replicon_type"]
                          .value_counts().items()},
        "ori_position_known": float(gene_table["ori_position"].notna()
                                    .mean()),
        "complete_assemblies": int(gtab["assembly_complete"].sum()),
        "operon_size": gene_table.drop_duplicates("operon_id")["operon_size"]
        .describe().round(2).to_dict(),
        "condition_levels_raw_clean": levels,
        "obs": obs_checks,
    }
    ecoli = gtab[gtab["genome_id"] == "GCF_000005845.2"]
    if len(ecoli):
        report["ecoli_k12_ori_ter"] = [int(ecoli["ori"].iloc[0]),
                                       int(ecoli["ter"].iloc[0])]
    (out / "report.json").write_text(json.dumps(report, indent=2,
                                                default=str))
    log(f"report: {json.dumps(report, default=str)}")
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
