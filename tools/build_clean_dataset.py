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

Sequence signals known to drive expression, computed per gene:

- ``sd_score`` (0 to 6 bases of AGGAGG) and ``sd_spacing``: the best
  Shine-Dalgarno site 20 to 9 bp before the start codon;
- ``sigma70_score`` (0 to 12), ``sigma70_m35``, ``sigma70_m10`` and
  ``sigma70_dist``: the best TTGACA / TATAAT pair 15 to 19 bp apart in the
  upstream sequence (a sigma 70 promoter);
- ``cai``: codon adaptation index against the ribosomal proteins of the same
  genome (``is_ribosomal_protein``), the classic proxy of expression level;
- ``n_copies``: genes of the same genome with the very same sequence (their
  reads split between the copies).

Validation folds that mimic the held-out regions (``N_FOLDS`` folds):

- ``gene_fold`` by ``gene_family``, the genes linked by an ortholog group or
  a shared protein id (region C never shares either with training);
- ``run_fold`` by experiment, so replicates stay together (region B never
  shares an experiment with training);
- ``pair_fold`` in the observations, a hash of the (gene, run) pair (A).

Run quality from the training counts, to weight or drop outliers at fit time:
``qc_n_genes``, ``qc_zero_frac``, ``qc_top_frac`` (share of the reads on the
top gene), ``qc_corr`` (correlation with the mean profile of the genome) and
``qc_outlier``. They are NaN for runs absent from training.

The raw ``Chr``, ``Start``, ``End`` and ``Strand`` stay in
feature_annotation.csv, at the same row as ``gene_idx``.

    python tools/build_clean_dataset.py --mode train --workers 6
"""

import argparse
import json
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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
N_FOLDS = 5
RIBOSOMAL = re.compile(r"ribosomal (?:subunit )?protein [LS]\d")
NOT_RIBOSOMAL = re.compile(r"ase\b|-like")
# A run is an outlier when its profile barely follows its genome's mean
# profile, or when most of its genes got no read.
QC_MIN_CORR = 0.5
QC_MAX_ZERO = 0.5
BASE_CODE = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    BASE_CODE[ord(_b)] = BASE_CODE[ord(_b.lower())] = _i


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


def motif_matches(arr, motif):
    """``out[i, p]`` = bases of ``motif`` matched at position p of row i."""
    width = arr.shape[1] - len(motif) + 1
    out = np.zeros((arr.shape[0], width), dtype=np.int8)
    for k, base in enumerate(motif.encode()):
        out += arr[:, k:k + width] == base
    return out


def upstream_signals(upstreams, size):
    """Shine-Dalgarno and sigma 70 promoter scores of upstream sequences.

    Each upstream sequence is right aligned, so its last base is the one just
    before the start codon. Missing bases are N and never match.
    """
    n = len(upstreams)
    arr = np.full((n, size), ord("N"), dtype=np.uint8)
    for i, s in enumerate(upstreams):
        b = (s or "").upper().encode("ascii", "replace")[-size:]
        if b:
            arr[i, size - len(b):] = np.frombuffer(b, dtype=np.uint8)

    # Shine-Dalgarno AGGAGG starting 20 to 9 bp before the start codon
    sd = motif_matches(arr[:, size - 20:size - 3], "AGGAGG")
    sd_best = sd.argmax(axis=1)
    sd_score = sd[np.arange(n), sd_best]
    sd_spacing = 14 - sd_best            # bp between the site and the start

    # Sigma 70: -35 box TTGACA, 15 to 19 bp spacer, -10 box TATAAT
    m35 = motif_matches(arr, "TTGACA")
    m10 = motif_matches(arr, "TATAAT")
    width = m10.shape[1]
    best = np.full(n, -1, dtype=np.int16)
    best10 = np.zeros(n, dtype=np.int16)
    best35 = np.zeros(n, dtype=np.int16)
    pos10 = np.zeros(n, dtype=np.int64)
    rows = np.arange(n)
    for spacer in range(15, 20):
        shift = 6 + spacer
        comb = m35[:, :width - shift].astype(np.int16) + m10[:, shift:]
        arg = comb.argmax(axis=1)
        val = comb[rows, arg]
        better = val > best
        best[better] = val[better]
        pos10[better] = arg[better] + shift
        best10[better] = m10[rows, arg + shift][better]
        best35[better] = m35[rows, arg][better]
    return {"sd_score": sd_score.astype(np.int8),
            "sd_spacing": sd_spacing.astype(np.int8),
            "sigma70_score": best.astype(np.int8),
            "sigma70_m35": best35.astype(np.int8),
            "sigma70_m10": best10.astype(np.int8),
            # bp between the end of the -10 box and the start codon
            "sigma70_dist": (size - (pos10 + 6)).astype(np.int16)}


def codon_adaptation(cds_list, reference, table_id):
    """Codon adaptation index of each gene against the ``reference`` genes.

    A codon's weight is its count in the reference set divided by the count of
    the most used synonymous codon. The CAI of a gene is the geometric mean of
    the weights of its codons, start and stop codons left out.
    """
    from Bio.Data.CodonTable import unambiguous_dna_by_id

    n = len(cds_list)
    if reference.sum() < 10:
        return np.full(n, np.nan)
    families = {}
    for codon, aa in unambiguous_dna_by_id[table_id].forward_table.items():
        idx = 16 * "ACGT".index(codon[0]) + 4 * "ACGT".index(codon[1]) \
            + "ACGT".index(codon[2])
        families.setdefault(aa, []).append(idx)

    lengths = np.array([len(c) // 3 for c in cds_list], dtype=np.int64)
    joined = "".join(c[:3 * k] for c, k in zip(cds_list, lengths))
    bases = BASE_CODE[np.frombuffer(joined.encode("ascii", "replace"),
                                    dtype=np.uint8)].reshape(-1, 3)
    valid = (bases < 4).all(axis=1)
    code = np.where(valid, bases[:, 0].astype(np.int64) * 16
                    + bases[:, 1] * 4 + bases[:, 2], 0)
    gene = np.repeat(np.arange(n), lengths)
    ends = np.cumsum(lengths)
    keep = valid.copy()
    has = lengths > 0
    keep[(ends - lengths)[has]] = False      # start codon
    keep[(ends - 1)[has]] = False            # stop codon

    counts = np.bincount(code[keep & reference[gene]], minlength=64) + 0.5
    log_w = np.full(64, np.nan)
    for idx in families.values():
        if len(idx) > 1:
            idx = np.array(idx)
            log_w[idx] = np.log(counts[idx] / counts[idx].max())
    use = keep & ~np.isnan(log_w[code])
    total = np.bincount(gene[use], weights=log_w[code[use]], minlength=n)
    k = np.bincount(gene[use], minlength=n)
    return np.where(k > 0, np.exp(total / np.maximum(k, 1)), np.nan)


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

    # Sequence signals, codon usage and identical copies
    signals = upstream_signals(seq_rows["upstream_seq"], upstream)
    products = pd.Series(text["product"], dtype=object).fillna("")
    ribosomal = (products.str.contains(RIBOSOMAL)
                 & ~products.str.contains(NOT_RIBOSOMAL)).to_numpy()
    hypothetical = products.str.contains("hypothetical protein",
                                         regex=False).to_numpy()
    tables = Counter(int(f.qualifiers.get("transl_table", ["11"])[0])
                     for f in index.values() if f.type == "CDS")
    table_id = tables.most_common(1)[0][0] if tables else 11
    cds_upper = [(c or "").upper() for c in seq_rows["cds_seq"]]
    cai = codon_adaptation(cds_upper, ribosomal, table_id)
    copies = Counter(cds_upper)
    n_copies = np.array([copies[c] if c else 0 for c in cds_upper],
                        dtype=np.int16)

    features = pd.DataFrame({
        "gene_idx": gene_idx,
        **signals,
        "cai": cai,
        "n_copies": n_copies,
        "is_ribosomal_protein": ribosomal,
        "is_hypothetical": hypothetical,
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
        "transl_table": table_id,
        "cai_reference_genes": int(ribosomal.sum()),
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


def fold_of(groups, k, seed):
    """Spread groups over ``k`` folds, every member of a group in one fold."""
    perm = np.random.default_rng(seed).permutation(int(groups.max()) + 1)
    return (perm[groups] % k).astype(np.int8)


def gene_families(ortho_group, protein_group):
    """Connected components of the genes linked by orthology or protein id."""
    n = len(ortho_group)
    o = np.asarray(ortho_group, dtype=np.int64)
    p = np.asarray(protein_group, dtype=np.int64)
    n_o = int(o.max()) + 1
    size = n + n_o + int(p.max()) + 1
    rows = np.r_[np.arange(n), np.arange(n)]
    cols = np.r_[n + o, n + n_o + p]
    graph = coo_matrix((np.ones(2 * n, dtype=np.int8), (rows, cols)),
                       shape=(size, size))
    _, labels = connected_components(graph, directed=False)
    return pd.factorize(labels[:n])[0].astype(np.int32)


def pair_fold(gene_idx, run_idx, k):
    """Deterministic fold of each (gene, run) pair, from a mixed hash."""
    h = (gene_idx.astype(np.uint64) * np.uint64(2654435761)
         + run_idx.astype(np.uint64) * np.uint64(40503))
    h ^= h >> np.uint64(15)
    h *= np.uint64(2246822519)
    h ^= h >> np.uint64(13)
    return (h % np.uint64(k)).astype(np.int8)


def run_qc(gene_idx, run_idx, count, log_expr, n_genes, n_runs):
    """Quality of each run, and the mean log expression of each gene.

    ``qc_corr`` correlates the run's log expression with the mean log
    expression of the same genes over all training runs.
    """
    n = np.bincount(run_idx, minlength=n_runs).astype(np.float64)
    zeros = np.bincount(run_idx, weights=(count == 0), minlength=n_runs)
    total = np.bincount(run_idx, weights=count, minlength=n_runs)
    top = np.zeros(n_runs, dtype=np.int64)
    np.maximum.at(top, run_idx, count)
    gene_n = np.bincount(gene_idx, minlength=n_genes)
    gene_mean = (np.bincount(gene_idx, weights=log_expr, minlength=n_genes)
                 / np.maximum(gene_n, 1))
    m = gene_mean[gene_idx]
    x = log_expr.astype(np.float64)
    sums = {name: np.bincount(run_idx, weights=w, minlength=n_runs)
            for name, w in (("x", x), ("m", m))}
    sxx = np.bincount(run_idx, weights=x * x, minlength=n_runs)
    smm = np.bincount(run_idx, weights=m * m, minlength=n_runs)
    sxm = np.bincount(run_idx, weights=x * m, minlength=n_runs)
    del m, x
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, mm = sums["x"] / n, sums["m"] / n
        cov = sxm / n - mx * mm
        corr = cov / np.sqrt((sxx / n - mx ** 2) * (smm / n - mm ** 2))
        seen = n > 0
        qc = pd.DataFrame({
            "qc_n_genes": np.where(seen, n, np.nan).astype(np.float32),
            "qc_zero_frac": np.where(seen, zeros / n, np.nan)
            .astype(np.float32),
            "qc_top_frac": np.where(seen, top / total, np.nan)
            .astype(np.float32),
            "qc_corr": np.where(seen, corr, np.nan).astype(np.float32),
        })
    qc["qc_outlier"] = pd.array(
        np.where(seen, (qc["qc_corr"] < QC_MIN_CORR)
                 | (qc["qc_zero_frac"] > QC_MAX_ZERO), pd.NA),
        dtype="boolean")
    gene_mean = np.where(gene_n > 0, gene_mean, np.nan)
    return qc, gene_mean


def build_obs(src, out, split, genes, runs, llr, lsf):
    """Write obs_<split>.parquet and check it against the source rows.

    For the train split, also returns the run quality table and the mean log
    expression of each gene (used by the report only, never stored).
    """
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
    folds = pair_fold(gene_idx, run_idx, N_FOLDS)
    check["pair_fold_sizes"] = np.bincount(folds, minlength=N_FOLDS).tolist()
    table = pa.table({"gene_idx": gene_idx, "run_idx": run_idx,
                      "readcount": count, "offset": offset,
                      "log_expr": log_expr, "pair_fold": folds})
    pq.write_table(table, out / f"obs_{split}.parquet", compression="zstd",
                   row_group_size=4_000_000)
    del table, folds, offset
    log(f"obs_{split}: {check}")
    if split != "train":
        return check, None, None
    qc, gene_mean = run_qc(gene_idx, run_idx, count, log_expr, len(genes),
                           len(runs))
    return check, qc, gene_mean


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
    clean["gene_family"] = gene_families(clean["ortho_group"],
                                         clean["protein_group"])
    clean["gene_fold"] = fold_of(clean["gene_family"].to_numpy(), N_FOLDS,
                                 seed=0)
    gene_table = pd.concat([clean, feats], axis=1)
    gene_table.to_parquet(out / "genes.parquet", index=False)
    log(f"genes.parquet: {gene_table.shape}")

    # 4. Run table, folds by experiment so that replicates stay together
    run_table, levels = clean_runs(runs, genome_idx)
    exp = run_table["exp_idx"].to_numpy()
    run_group = pd.factorize(np.where(
        exp >= 0, exp, exp.max() + 1 + run_table["run_idx"].to_numpy()))[0]
    run_table["run_fold"] = fold_of(run_group, N_FOLDS, seed=1)

    # 5. Observations, then the run quality measured on the train counts
    llr = genes["log_length_ratio"].to_numpy()
    lsf = runs["log_size_factor"].to_numpy()
    obs_checks, qc, gene_mean = {}, None, None
    for split in SPLITS:
        if (src / f"{split}.parquet").exists():
            check, split_qc, split_mean = build_obs(
                src, out, split, genes, run_table, llr, lsf)
            obs_checks[split] = check
            if split_qc is not None:
                qc, gene_mean = split_qc, split_mean
    if qc is not None:
        run_table = pd.concat([run_table, qc], axis=1)
    run_table.to_parquet(out / "runs.parquet", index=False)
    log(f"runs.parquet: {run_table.shape}, levels {levels}")

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
        "gene_families": int(gene_table["gene_family"].max() + 1),
        "largest_family": int(np.bincount(gene_table["gene_family"]).max()),
        "gene_fold_sizes": np.bincount(gene_table["gene_fold"],
                                       minlength=N_FOLDS).tolist(),
        "run_fold_sizes": np.bincount(run_table["run_fold"],
                                      minlength=N_FOLDS).tolist(),
        "cai_reference_genes": gtab["cai_reference_genes"].describe()
        .round(1).to_dict(),
        "n_copies_above_1": int((gene_table["n_copies"] > 1).sum()),
        "obs": obs_checks,
    }
    if qc is not None:
        report["qc_corr_quantiles"] = run_table["qc_corr"].astype(float)\
            .quantile([0.01, 0.05, 0.25, 0.5]).round(3).to_dict()
        report["qc_zero_frac_quantiles"] = run_table["qc_zero_frac"]\
            .astype(float).quantile([0.5, 0.95, 0.99]).round(3).to_dict()
        report["qc_outliers"] = int(run_table["qc_outlier"].sum())
        # Spearman correlation of each gene signal with the gene's mean log
        # expression over the training runs (a sanity check, not stored)
        target = pd.Series(gene_mean)
        report["spearman_with_mean_expression"] = {
            col: round(float(gene_table[col].astype(float).corr(
                target, method="spearman")), 3)
            for col in ("cai", "gc3", "sd_score", "sigma70_score",
                        "operon_rank", "ori_position", "n_copies", "length")}
        rib = gene_table["is_ribosomal_protein"].to_numpy()
        report["mean_expression_ribosomal_vs_other"] = [
            round(float(np.nanmean(gene_mean[rib])), 3),
            round(float(np.nanmean(gene_mean[~rib])), 3)]
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
