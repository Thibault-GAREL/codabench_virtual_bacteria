"""Expression transformer: one row (a gene in a run) read as a set of tokens.

    tokens = [CLS] gene  ESM+signals  genome  run  project  condition
             medium  oxygen  growth phase ...  (one token per curated field)
             temperature  pH  agitation  OD600 (one token per number)
    log mu    = offset + id biases + head_mu(CLS)
    log sigma = rho0 + rho_gene + head_r(CLS) + delta_case

- The id tokens (gene, run, project, condition) are free embeddings. A level
  the model has never seen reads a learned MASK token instead, so the model
  falls back on what it can see: the protein (ESM-2), the gene signals, the
  genome and the curation sheet of the run.
- Level dropout teaches that fallback: at each step, 1 gene family in 10,
  1 condition in 10, 1 project in 20 and 1 run in 20 lose their id tokens,
  as in the regions C (new gene) and B (new run).
- The id biases (``id_bias``) are per-level additive effects: one free value
  per gene, genome, run and project, as a GLM would have, but learned inside
  the network and started from moment estimates. Off in the submitted
  preset: there is no GLM in that model. A separate, frozen GLM only comes
  back with ``glm_eta``.
- ``mixer=mlp`` replaces the attention blocks by one residual MLP over the
  flattened tokens, to measure what the attention brings. ``out_scale`` puts
  the head output of log mu in units of the spread of the target.
- ``count_stats`` adds statistics of the training counts as inputs, one
  token each (``_CountStats``): the mean level of the run, of the gene, of
  the gene in the project and in the condition, and the level of the two
  genome neighbours of the gene in the same run. A learned embedding needs
  many steps to learn a gene's level, these means are exact from the start.
  None of them ever reads the row's own count: each training row only sees
  the folds its block (gene x project and condition) is not in, the way the
  region A hides its rows. ``stats_sd`` adds the spread of the gene in its
  project and condition, ``a_holdout=block`` builds the internal holdout of
  region A from the same blocks, so that sigma is calibrated on a case as
  hard as region A.
- Every component of the 2017 Transformer that Modern_LLM replaces is a flag
  of ``TConfig``, off by default, so each one can be measured alone
  (``tools/ablate.py``): pre-norm, RMSNorm, DyT, SwiGLU, GeGLU, GQA, MQA,
  QK-norm, logit softcapping, fused attention (SDPA), sparse MoE, GPT init with
  scaled residuals, AdamW without decay on norms and embeddings, gradient
  clipping, cosine or WSD schedule, fp16 or bf16. Positional encodings (RoPE,
  ALiBi) are left out: the tokens are a set, each one has its own slot
  embedding instead of a position.

Fit, on the GPU: holdouts built like the regions (gene families, conditions,
experiments and projects, gene x condition blocks, pairs), training on the
other rows with train and validation curves, the best checkpoint kept, one
dispersion shift per case calibrated on the holdouts with the challenge
metric, then a short refit on every row. A time budget (``max_minutes``)
stops each phase early so that train and predict fit in the hour of the
platform.
"""

import gc as _gc
import math
import os
import re
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from benchmark_utils.base_solver import GeneCompetSolver
from benchmark_utils.nb_loss import NLL_CAP, nb_nll

CHUNK = 8_000_000
EVAL_CHUNK = 32_768
CONDITION_FIELDS = [
    "is_wild_type", "medium_category", "medium_name", "temperature_celsius",
    "ph", "oxygen_condition", "agitation_rpm", "growth_mode", "growth_phase",
    "od600", "has_perturbation", "perturbation_category",
    "perturbation_compound"]
CASES = ["A_full", "A_block", "B_condition", "B_project", "B_new", "C", "D"]
# Case whose shift is used when a case has too few holdout rows
FALLBACK = {"A_full": "A_block", "A_block": "A_full",
            "B_condition": "B_project", "B_project": "B_new",
            "B_new": "B_project", "C": "A_block", "D": "C"}
MIN_CALIB_ROWS = 2000
MAX_CALIB_ROWS = 1_000_000
FEATURE_FILE = "gene_features.npz"
# Environment variable that points at the feature file in local runs
FEATURE_ENV = "BACTERIA_GENE_FEATURES"
RUN_CATEGORIES = ["medium_category", "oxygen_condition", "growth_mode",
                  "growth_phase", "perturbation_category", "is_wild_type",
                  "has_perturbation", "condition_confidence"]
RUN_NAMES = ["medium_name", "perturbation_compound"]
RUN_NUMBERS = ["temperature_celsius", "ph", "agitation_rpm", "od600"]
TOP_NAMES = 40
# Rows of the fixed samples that draw the train and validation curves
CURVE_ROWS = 100_000
# Prior standard deviation of each effect, on the log scale, as in the GLM.
# GLM_PRIOR_SD for the frozen GLM of glm_eta, BIAS_PRIOR_SD for the id biases
# of the network (bias_prior).
GLM_PRIOR_SD = {"a_genome": 3.0, "a_gene": 3.0, "b_project": 1.0,
                "b_condition": 1.0, "b_run": 1.0, "rho_genome": 1.0,
                "rho_gene": 1.0}
BIAS_PRIOR_SD = {"a_genome": 3.0, "a_gene": 3.0, "b_project": 1.0,
                 "b_run": 1.0, "rho_gene": 1.0}
# Rows per forward / backward pass of the GLM fit
GPU_CHUNK = 2_000_000
# Prepared tables kept across fits of one process, see TransformerModel.prepare
KEEP_ENV = "BACTERIA_KEEP_PREPARED"
_PREPARED = {}
# Neighbour rows of the training rows, kept like _PREPARED (count_stats)
_NEIGHBOURS = {}
# The inputs of count_stats, the first N_MEAN_STATS without the neighbours.
# Means on z = log(y + 0.5) - offset, the gene ones on z minus the run mean.
STAT_NAMES = ["run_mean", "gene_mean", "gene_sd", "gene_rows", "prj_mean",
              "prj_rows", "cond_mean", "cond_rows", "up_level", "up_dev",
              "down_level", "down_dev"]
N_MEAN_STATS = 8
# with stats_sd, after the others: the spread of the gene's c in the project
# and in the condition, so sigma knows where the gene is unstable
SD_STAT_NAMES = ["prj_sd", "cond_sd"]


def _parse_shifts(text):
    """The sigma_extra string "A_full=-0.3,C=-0.15" as a dict of CASES."""
    out = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, value = (s.strip() for s in part.split("="))
        if name not in CASES:
            raise ValueError(f"unknown case {name} in sigma_extra")
        out[name] = float(value)
    return out


def _stat_names(count_stats, stats_sd):
    """Names of the count statistics of a configuration, in token order."""
    names = {"off": [], "means": STAT_NAMES[:N_MEAN_STATS],
             "all": STAT_NAMES}[count_stats]
    return names + (SD_STAT_NAMES if names and stats_sd else [])


@dataclass
class TConfig:
    """Every knob of the model. The defaults are the 2017 Transformer."""

    # size
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ffn_mult: float = 4.0
    dropout: float = 0.0
    # Modern_LLM flags, the 2017 choice first
    norm: str = "layernorm"        # layernorm | rmsnorm | dyt
    norm_place: str = "post"       # post | pre | sandwich
    ffn: str = "relu"              # relu | gelu | swiglu | geglu
    attn: str = "mha"              # mha | gqa | mqa
    n_kv_heads: int = 2            # gqa only
    qk_norm: bool = False
    softcap: float = 0.0           # 0 is off, else c * tanh(score / c)
    sdpa: bool = False             # fused attention kernel instead of softmax(QK)V
    init: str = "torch"            # torch | gpt (normal 0.02, residual outputs / sqrt(2L))
    moe_experts: int = 0           # 0 is a dense FFN
    moe_top_k: int = 2
    moe_aux: float = 0.01
    # optimisation
    optimizer: str = "adam"        # adam (betas 0.9, 0.98) | adamw (0.9, 0.95)
    lr: float = 1e-3
    weight_decay: float = 0.1      # adamw only, never on norms, biases, embeddings
    grad_clip: float = 0.0         # 0 is off
    schedule: str = "noam"         # noam | cosine | wsd | constant
    warmup_steps: int = 200
    precision: str = "fp32"        # fp32 | fp16 | bf16
    # data and budget
    batch: int = 32768
    epochs: float = 1.0
    refit_epochs: float = 0.25
    max_minutes: float = 40.0      # fit and refit together, the hour of the platform
    eval_every: int = 100
    keep_best: bool = True
    # domain
    id_bias: bool = True
    drop_gene: float = 0.1
    drop_cond: float = 0.1
    drop_prj: float = 0.05
    drop_run: float = 0.05
    seed: int = 0
    # the mean mu, tests of 2026-11-06 (all off in the submitted 7.11)
    bias_prior: float = 0.0        # L2 on the id biases, 1 is the GLM prior
    glm_eta: str = "off"           # off | offset | token: a frozen additive GLM
    glm_steps: int = 150
    glm_refit_steps: int = 60
    n_models: int = 1              # seeds, mixed by moments
    out_scale: bool = False        # head log mu in units of the target spread
    mixer: str = "attention"       # attention | mlp (no attention at all)
    mlp_hidden: int = 512          # width of the mlp mixer
    num_embed: str = "linear"      # linear | plr: periodic embeddings of the numbers
    plr_freq: int = 8              # frequencies per number (plr)
    plr_dim: int = 8               # size of each gene signal embedding (plr)
    # statistics of the training counts as inputs (2026-11-07)
    count_stats: str = "off"       # off | means | all (means + neighbours)
    stats_folds: int = 4           # folds of blocks, 1 hidden from each row
    stats_sd: bool = False         # + spread of the gene in project, condition
    # internal holdout of region A: pair (until 2026-11-07, condition blocks
    # and random (gene, run) pairs whose replicates stay visible) | block
    # (whole gene x project and condition blocks, as region A hides them)
    a_holdout: str = "pair"
    # share of the (gene, project) pairs hidden in their whole project, in
    # the folds of count_stats and in the block holdout (_gene_blocks)
    project_share: float = 0.0
    # extra shift of log sigma per case, on top of the calibration on the
    # internal holdouts: "A_full=-0.3,C=-0.15". The internal holdouts are
    # easier than the regions, so sigma comes out too tight everywhere: the
    # shifts are measured on the local regions of the train release
    sigma_extra: str = ""


# The configurations this file can submit, on top of the 2017 defaults, each
# measured at the T4 hour of the platform (local challenge NB-NLL, and the
# platform score once submitted). PRESET is the one the Solver uses, and
# tools/build_submission_zip.py builds the zip of any of them by changing
# the SUBMITTED line alone.
_SOCLE = {"norm_place": "pre", "norm": "rmsnorm", "ffn": "swiglu",
          "optimizer": "adamw", "grad_clip": 1.0, "schedule": "cosine",
          "precision": "fp16", "sdpa": False, "id_bias": False}
PRESETS = {
    # 2026-11-05: the modern socle without the per-level effects, in fp16
    # without the fused attention. Platform 7.11.
    "transformer": {**_SOCLE, "epochs": 3.0},
    # 2026-11-06 t4 suite: two transformers sharing the hour, mixed by
    # moments. Local 7.505, platform 7.03.
    "transformer_2seeds": {**_SOCLE, "epochs": 3.0, "n_models": 2},
    # 2026-11-06 mlp suite: three attention-free MLP 2x512 with dropout 0.1
    # sharing the hour (about 2 epochs each). Local 7.456, the best so far.
    "mlp_3seeds": {**_SOCLE, "mixer": "mlp", "dropout": 0.1, "epochs": 8.0,
                   "n_models": 3},
    # same suite: three wider MLP 2x1024 without dropout. Local 7.486.
    "mlp_3seeds_1024": {**_SOCLE, "mixer": "mlp", "mlp_hidden": 1024,
                        "epochs": 8.0, "n_models": 3},
    # 2026-11-07: mlp_3seeds with the statistics of the training counts as
    # inputs (count_stats). Not measured yet, the stats suite of ablate.py.
    "mlp_stats_3seeds": {**_SOCLE, "mixer": "mlp", "dropout": 0.1,
                         "epochs": 8.0, "n_models": 3, "count_stats": "all"},
    # + the spreads of the gene in its project and condition, and the
    # internal holdout of region A built like region A. The stats2 suite.
    "mlp_stats2_3seeds": {**_SOCLE, "mixer": "mlp", "dropout": 0.1,
                          "epochs": 8.0, "n_models": 3, "count_stats": "all",
                          "stats_sd": True, "a_holdout": "block"},
    # one model: with the statistics, three seeds of 2.4 epochs each do no
    # better than one of 7 epochs (local 7.411 against 7.393, 2026-11-07)
    "mlp_stats": {**_SOCLE, "mixer": "mlp", "dropout": 0.1, "epochs": 8.0,
                  "count_stats": "all"},
    # mlp_stats with sigma widened per case by half the shifts that fit the
    # local regions best (diagnostic of 2026-11-07 on the mlp_stats model:
    # A_full -0.63, A_block -0.37, B_condition -0.78, B_project -0.83, B_new
    # -0.50, C -0.29). On held-out halves of the local rows: 7.394 to 7.322
    # at half strength, 7.301 at full strength. Half, as the platform
    # regions look a little easier than the local ones.
    "mlp_stats_cal": {**_SOCLE, "mixer": "mlp", "dropout": 0.1,
                      "epochs": 8.0, "count_stats": "all",
                      "sigma_extra": "A_full=-0.32,A_block=-0.18,"
                                     "B_condition=-0.39,B_project=-0.41,"
                                     "B_new=-0.25,C=-0.15"},
    # + half of the genes hidden in their whole project in the folds and the
    # A holdout, as region A hides them. The stats3 suite.
    "mlp_stats3": {**_SOCLE, "mixer": "mlp", "dropout": 0.1, "epochs": 8.0,
                   "count_stats": "all", "a_holdout": "block",
                   "project_share": 0.5},
}
SUBMITTED = "mlp_3seeds"
PRESET = PRESETS[SUBMITTED]


# --- data helpers, shared with the GLM and the hybrid ----------------------

def _clean_name(value):
    """Lower case, no parenthesis, single spaces."""
    if not isinstance(value, str):
        return "unknown"
    value = re.sub(r"\([^)]*\)", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip() or "unknown"


def _gene_features(genes, submission_dir, log):
    """Gene inputs aligned on the gene table, zeros without the file."""
    paths = [Path(submission_dir) / FEATURE_FILE] if submission_dir else []
    if os.environ.get(FEATURE_ENV):
        paths.append(Path(os.environ[FEATURE_ENV]))
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        log(f"no {FEATURE_FILE}, the gene token sees nothing but its genome")
        return np.zeros((len(genes), 1), dtype=np.float16), 0
    z = np.load(path)
    keys = pd.Index(np.char.add(np.char.add(z["genome_id"], b"|"),
                                z["feature_id"]).astype(str))
    own = (genes["genome_id"].astype(str) + "|"
           + genes["feature_id"].astype(str))
    row = keys.get_indexer(own)
    found = row >= 0
    numeric = np.zeros((len(genes), z["numeric"].shape[1]), dtype=np.float16)
    numeric[found] = z["numeric"][row[found]]
    prot = np.full(len(genes), -1, dtype=np.int64)
    prot[found] = z["protein_idx"][row[found]]
    esm = z["esm"]
    emb = np.zeros((len(genes), esm.shape[1]), dtype=np.float16)
    has = prot >= 0
    emb[has] = esm[prot[has]]
    log(f"gene features {path.name}: {int(found.sum()):,} of {len(genes):,} "
        f"genes, {numeric.shape[1]} signals, ESM {esm.shape[1]} dimensions, "
        f"{int(has.sum()):,} with a protein")
    # the signals come first, so plr can take them apart from the ESM block
    x = np.concatenate([numeric, emb, has[:, None].astype(np.float16),
                        found[:, None].astype(np.float16)], axis=1)
    return x, numeric.shape[1]


def _run_tokens(runs):
    """Curated fields of each run as token codes, numbers as z-scores.

    Returns the codes (one column per field, offset into a single vocabulary),
    the vocabulary size, the numbers and their known flags.
    """
    codes, sizes = [], []
    for col in RUN_CATEGORIES + RUN_NAMES:
        if col not in runs:
            continue
        if col in RUN_NAMES:
            values = runs[col].map(_clean_name)
            top = values[values != "unknown"].value_counts().index[:TOP_NAMES]
            values = values.where(values.isin(top) | (values == "unknown"),
                                  "other")
        else:
            values = runs[col].map(
                lambda v: "unknown" if pd.isna(v) else str(v).lower())
        code, uniques = pd.factorize(values, sort=True)
        codes.append(code)
        sizes.append(len(uniques))
    if not codes:
        codes, sizes = [np.zeros(len(runs), dtype=np.int64)], [1]
    offsets = np.cumsum([0] + sizes[:-1])
    cat = (np.stack(codes, 1) + offsets).astype(np.int64)
    nums, known = [], []
    for col in RUN_NUMBERS:
        v = (pd.to_numeric(runs[col], errors="coerce") if col in runs
             else pd.Series(np.nan, index=runs.index))
        sd = v.std()
        z = (v - v.mean()) / (sd if sd > 0 else 1.0)
        nums.append(z.fillna(0.0).clip(-5, 5).to_numpy(np.float32))
        known.append(v.notna().to_numpy(np.float32))
    return cat, int(sum(sizes)), np.stack(nums, 1), np.stack(known, 1)


def _print_log(message):
    print(f"{time.strftime('%H:%M:%S')} [transformer] {message}", flush=True)


def _no_metric(step=None, show=False, **values):
    return None


def _ram():
    """Resident memory of the process, for the log."""
    try:
        import psutil
        return f"RAM {psutil.Process().memory_info().rss / 1e9:.1f} GB"
    except ImportError:
        return ""


def _recode(values, categories):
    """Integer code of each value in ``categories``, -1 when absent."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.cat.rename_categories(
            values.cat.categories.astype(str))
        return values.cat.set_categories(categories).cat.codes.to_numpy()
    return pd.Categorical(values.astype(str), categories=categories).codes


def _hit(groups, k, seed):
    """True for the groups that fall in the first of ``k`` random folds."""
    perm = np.random.default_rng(seed).permutation(int(groups.max()) + 1)
    return perm[groups] % k == 0


def _hash_hit(a, b, k):
    """True for 1 (a, b) pair in ``k``, from a mixed hash, chunk by chunk."""
    out = np.empty(len(a), dtype=bool)
    for s in range(0, len(a), CHUNK):
        h = (a[s:s + CHUNK].astype(np.uint64) * np.uint64(2654435761)
             + b[s:s + CHUNK].astype(np.uint64) * np.uint64(40503))
        h ^= h >> np.uint64(15)
        h *= np.uint64(2246822519)
        h ^= h >> np.uint64(13)
        out[s:s + CHUNK] = h % np.uint64(k) == 0
    return out


def _gene_families(genes):
    """Connected components of the genes linked by orthology or protein id."""
    own = genes["genome_id"].astype(str) + "|" + genes["feature_id"].astype(str)
    if "ref_feature_id" in genes:
        ref = genes["ref_feature_id"]
        ortho = np.where(ref.notna(), genes["ref_genome_id"].astype(str) + "|"
                         + ref.astype(str), own)
    else:
        ortho = own.to_numpy()
    o = pd.factorize(ortho)[0].astype(np.int64)
    p = pd.factorize(genes["feature_id"].astype(str))[0].astype(np.int64)
    n, n_o = len(o), int(o.max()) + 1
    size = n + n_o + int(p.max()) + 1
    graph = coo_matrix((np.ones(2 * n, dtype=np.int8),
                        (np.r_[np.arange(n), np.arange(n)],
                         np.r_[n + o, n + n_o + p])), shape=(size, size))
    return pd.factorize(connected_components(graph, directed=False)[1][:n])[0]


def _nb_nll_torch(y, log_mu, log_r):
    """NB negative log-likelihood without the constant lgamma(y + 1)."""
    r = torch.exp(log_r)
    log_rm = torch.logaddexp(log_r, log_mu)
    ll = (torch.lgamma(y + r) - torch.lgamma(r)
          + r * (log_r - log_rm) + y * (log_mu - log_rm))
    return -ll


def _nb_nll_capped(y, log_mu, log_r):
    """Per row challenge metric: full NLL capped at NLL_CAP, as predicted."""
    log_mu = log_mu.clamp(-12, 18)
    log_r = log_r.clamp(-7, 12)
    full = _nb_nll_torch(y, log_mu, log_r) + torch.lgamma(y + 1)
    return full.clamp(max=NLL_CAP)


def _mix(etas, log_rs):
    """One NB with the mean and the variance of the models' NB mixture.

    Equal weights. The mixture of NB is not a NB: this keeps its first two
    moments, so a disagreement between the models widens sigma.
    """
    if len(etas) == 1:
        return etas[0], log_rs[0]
    mu = torch.stack([e.clamp(-12, 18).exp() for e in etas]).double()
    r = torch.stack([v.clamp(-7, 12).exp() for v in log_rs]).double()
    m = mu.mean(0)
    second = (mu + mu ** 2 / r + mu ** 2).mean(0)
    # variance beyond Poisson, positive by Jensen
    excess = (second - m ** 2 - m).clamp(min=1e-12)
    log_m = torch.log(m)
    return log_m.float(), (2 * log_m - torch.log(excess)).float()


class _Index:
    """Integer ids of genes, genomes, runs, projects and conditions."""

    def __init__(self, genes, runs):
        self.genome_cats = pd.Index(sorted(genes["genome_id"].astype(str)
                                           .unique()))
        self.feature_cats = pd.Index(genes["feature_id"].astype(str).unique())
        g = _recode(genes["genome_id"], self.genome_cats).astype(np.int64)
        f = _recode(genes["feature_id"], self.feature_cats).astype(np.int64)
        self.width = len(self.feature_cats) + 1
        code = g * self.width + f
        self.order = np.argsort(code)
        self.sorted_code = code[self.order]
        self.genome_of_gene = g
        self.n_gene, self.n_genome = len(genes), len(self.genome_cats)

        acc = runs["run_accession"].astype(str)
        self.run_cats = pd.Index(acc)
        self.n_run = len(runs)
        self.genome_of_run = _recode(runs["genome_id"],
                                     self.genome_cats).astype(np.int64)

        def ids(column):
            values = runs[column] if column in runs else pd.Series(
                [None] * len(runs))
            return pd.factorize(np.where(values.isna(), "run:" + acc,
                                         values.astype(str)))[0]

        self.prj_of_run = ids("prj_accession").astype(np.int32)
        self.exp_of_run = ids("experiment_accession").astype(np.int32)
        self.n_prj = int(self.prj_of_run.max()) + 1

        fields_ = [c for c in CONDITION_FIELDS if c in runs]
        if "condition_confidence" in runs:
            sheet = runs["condition_confidence"].notna().to_numpy()
        else:
            sheet = runs[fields_].notna().any(axis=1).to_numpy()
        key = runs["genome_id"].astype(str)
        for column in fields_:
            # not astype(str): pandas 3 keeps a missing value missing, and a
            # single empty field would then blank the whole key
            key = key + "|" + runs[column].map(
                lambda v: "nan" if pd.isna(v) else str(v))
        self.has_sheet = sheet
        self.cond_of_run = pd.factorize(key.where(sheet))[0].astype(np.int32)
        self.n_cond = max(int(self.cond_of_run.max()) + 1, 1)

    def rows(self, df):
        """gene, genome and run id of each row (-1 when unknown)."""
        g_all = _recode(df["genome_id"], self.genome_cats)
        f_all = _recode(df["feature_id"], self.feature_cats)
        gene = np.empty(len(df), dtype=np.int32)
        for s in range(0, len(df), CHUNK):
            g = g_all[s:s + CHUNK].astype(np.int64)
            f = f_all[s:s + CHUNK].astype(np.int64)
            code = g * self.width + f
            pos = np.minimum(np.searchsorted(self.sorted_code, code),
                             len(self.sorted_code) - 1)
            found = (self.sorted_code[pos] == code) & (g >= 0) & (f >= 0)
            gene[s:s + CHUNK] = np.where(found, self.order[pos], -1)
        run = _recode(df["run_accession"], self.run_cats).astype(np.int32)
        return gene, g_all.astype(np.int32), run


class _Pairs:
    """Factorised (gene, group) pairs, with a lookup for new rows."""

    def __init__(self, gene, group, n_group):
        self.width = n_group + 1
        code = gene.astype(np.int64) * self.width + group
        valid = group >= 0
        if valid.all():
            idx, uniques = pd.factorize(code)
            self.idx = idx.astype(np.int32)
        else:
            self.idx = np.full(len(gene), -1, dtype=np.int32)
            idx, uniques = pd.factorize(code[valid])
            self.idx[valid] = idx
        code = idx = None
        self.order = np.argsort(uniques)
        self.sorted = uniques[self.order]
        self.n = len(uniques)

    def lookup(self, gene, group):
        code = gene.astype(np.int64) * self.width + group
        pos = np.minimum(np.searchsorted(self.sorted, code),
                         max(len(self.sorted) - 1, 0))
        found = ((self.sorted[pos] == code) & (gene >= 0) & (group >= 0)
                 if self.n else np.zeros(len(code), dtype=bool))
        return np.where(found, self.order[pos] if self.n else -1, -1)


# --- statistics of the training counts (count_stats) ------------------------

def _neighbours(genes):
    """Upstream and downstream gene of each gene on its replicon, -1 if none.

    Upstream is the 5' side in the gene's own reading direction: the gene
    before it on the + strand, the gene after it on the - strand. A split
    gene (coordinates "a;b") starts at its first part. Also returns whether
    each neighbour reads on the same strand, as an operon partner does.
    """
    n = len(genes)
    none = np.full(n, -1, dtype=np.int64)
    if not {"Chr", "Start", "Strand"} <= set(genes.columns):
        return none, none.copy(), np.zeros(n, bool), np.zeros(n, bool)
    # a copy: pandas 3 hands out read-only arrays, and split genes are
    # filled in below
    start = pd.to_numeric(genes["Start"], errors="coerce").to_numpy(
        dtype=float, copy=True)
    for i in np.flatnonzero(np.isnan(start)):
        parts = re.findall(r"\d+", str(genes["Start"].iat[i]))
        start[i] = min(float(p) for p in parts) if parts else np.nan
    chrom = (genes["genome_id"].astype(str) + "|"
             + genes["Chr"].astype(str).str.split(";").str[0])
    code = pd.factorize(chrom)[0]
    plus = genes["Strand"].astype(str).str[:1].to_numpy() == "+"
    ok = ~np.isnan(start)
    order = np.lexsort((np.where(ok, start, 0.0), code))
    order = order[ok[order]]
    same = code[order[1:]] == code[order[:-1]]
    prev, nxt = none.copy(), none.copy()
    prev[order[1:][same]] = order[:-1][same]
    nxt[order[:-1][same]] = order[1:][same]
    up = np.where(plus, prev, nxt)
    down = np.where(plus, nxt, prev)

    def same_strand(nb):
        return (nb >= 0) & (plus[np.maximum(nb, 0)] == plus)

    return up, down, same_strand(up), same_strand(down)


def _stat_blocks(ix):
    """Block of each run for the folds: its project and condition, or its
    experiment when it has no sheet. Region A hides a gene in a whole block:
    12 % of its rows have the gene seen in the same project and condition,
    none in the same experiment (measured on the train mode, 2026-11-07)."""
    code = np.where(ix.cond_of_run >= 0,
                    ix.prj_of_run.astype(np.int64) * (ix.n_cond + 1)
                    + ix.cond_of_run, -1 - ix.exp_of_run.astype(np.int64))
    return pd.factorize(code)[0].astype(np.int64)


def _gene_blocks(gene, run, ix, project_share=0.0):
    """Block of each training row, for the folds and the A holdout.

    The project and condition block of the run (_stat_blocks), or for a
    share of the (gene, project) pairs the whole project. Region A sees the
    gene in the same project for 40.5 % of its rows only: hiding gene x
    (project, condition) blocks leaves it seen for 84.5 % of the hidden
    rows, hiding half of the genes in their whole project for 42 %
    (measured on the train mode, 2026-11-07).
    """
    blocks = _stat_blocks(ix)
    block = blocks[run]
    if project_share <= 0:
        return block
    prj = ix.prj_of_run[run].astype(np.int64)
    whole = _hash_fold(gene, prj + 10**9, 100) < round(100 * project_share)
    return np.where(whole, int(blocks.max()) + 1 + prj, block)


def _hash_fold(a, b, k):
    """Fold in [0, k) of each (a, b) pair, a hash unrelated to _hash_hit."""
    out = np.empty(len(a), dtype=np.int8)
    for s in range(0, len(a), CHUNK):
        h = (a[s:s + CHUNK].astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
             + b[s:s + CHUNK].astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F))
        h ^= h >> np.uint64(31)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(29)
        out[s:s + CHUNK] = (h % np.uint64(k)).astype(np.int8)
    return out


def _row_lookup(gene, run, n_run):
    """Keys gene * n_run + run of the training rows, sorted, and their rows."""
    key = gene.astype(np.int64) * n_run + run
    order = np.argsort(key, kind="stable")
    return key[order], order.astype(np.int32)


def _find_rows(sorted_key, order, of_gene, gene, run, n_run):
    """Training row of the pair (of_gene[gene], run), -1 when absent."""
    out = np.full(len(gene), -1, dtype=np.int32)
    if not len(sorted_key):
        return out
    for s in range(0, len(gene), CHUNK):
        g = gene[s:s + CHUNK].astype(np.int64)
        r = run[s:s + CHUNK].astype(np.int64)
        nb = np.where(g >= 0, of_gene[np.maximum(g, 0)], -1)
        key = nb * n_run + r
        pos = np.minimum(np.searchsorted(sorted_key, key), len(sorted_key) - 1)
        hit = (nb >= 0) & (r >= 0) & (sorted_key[pos] == key)
        out[s:s + CHUNK] = np.where(hit, order[pos], -1)
    return out


class _CountStats:
    """Statistics of the training counts as inputs, never of the row itself.

    On z = log(y + 0.5) - offset, from the visible training rows:

    - ``run_mean``: mean z of the run, its level over all its genes;
    - ``gene_mean``, ``gene_sd``, ``gene_rows``: the gene on z minus the run
      mean (c), its mean, spread and log1p row count over every run;
    - ``prj_mean``, ``cond_mean`` (and their row counts): the mean c of the
      gene in the same project, in the same condition;
    - ``up_level``, ``down_level``: z of the two genome neighbours of the
      gene in the same run, minus the run mean;
    - ``up_dev``, ``down_dev``: the same z minus the neighbour's own mean z,
      how much more its operon is on in this run than usual;
    - with ``sd``: ``prj_sd``, ``cond_sd``, the spread of c of the gene in
      the project and in the condition.

    No leakage: every training row is in one of ``k`` folds, drawn per
    (gene, block) as region A hides its rows (see _stat_blocks). A training
    row only sums the other folds, so its own count, and those of its whole
    block, never enter its statistics, and a neighbour of its own fold is
    hidden. The neighbour's own mean uses raw z, not c: c is centred by a
    run mean that holds the row. A row outside the visible ones (a holdout,
    a test row) reads every visible row, as a test row will.
    """

    def __init__(self, rows, n_gene, n_run, n_gp, n_gc, k, neighbours,
                 device, sd=False):
        # device tensors over the training rows: gene, run, y, off, gp, gc,
        # fold, nb_up, nb_down
        self.r = rows
        self.k = k
        self.size = {"run": n_run, "gene": n_gene, "gp": max(n_gp, 1),
                     "gc": max(n_gc, 1)}
        self.neighbours = neighbours
        self.sd = sd
        self.n_stats = len(_stat_names("all" if neighbours else "means", sd))
        self.device = device
        self.center = torch.zeros(self.n_stats, device=device)
        self.scale = torch.ones(self.n_stats, device=device)
        self.folds = torch.arange(k, device=device)
        self.vis = None

    def _z(self, i):
        return (torch.log(self.r["y"][i].float() + 0.5)
                - self.r["off"][i].float())

    @torch.no_grad()
    def build(self, vis):
        """Sums per level and per fold over the visible rows ``vis``."""
        K, dev, r = self.k, self.device, self.r
        self.vis = vis
        sel = torch.nonzero(vis).flatten()

        def table(name, n_sums):
            return [torch.zeros(self.size[name] * K, device=dev)
                    for _ in range(n_sums)]

        run = table("run", 2)
        for s in range(0, len(sel), GPU_CHUNK):
            i = sel[s:s + GPU_CHUNK]
            flat = r["run"][i].long() * K + r["fold"][i].long()
            run[0].index_add_(0, flat, self._z(i))
            run[1].index_add_(0, flat, torch.ones(len(i), device=dev))
        self.run = [v.view(-1, K) for v in run]
        run_mean = self.run[0].sum(1) / self.run[1].sum(1).clamp(min=1)
        # gene: sum c, sum c^2, rows, sum z (the neighbours' own mean). The
        # pairs: sum c, rows, sum c^2
        gene, gp, gc = table("gene", 4), table("gp", 3), table("gc", 3)
        for s in range(0, len(sel), GPU_CHUNK):
            i = sel[s:s + GPU_CHUNK]
            f = r["fold"][i].long()
            z = self._z(i)
            c = z - run_mean[r["run"][i].long()]
            one = torch.ones_like(c)
            flat = r["gene"][i].long() * K + f
            for v, x in zip(gene, (c, c * c, one, z), strict=True):
                v.index_add_(0, flat, x)
            for pairs, idx in ((gp, r["gp"][i].long()), (gc, r["gc"][i].long())):
                ok = idx >= 0
                flat = idx[ok] * K + f[ok]
                pairs[0].index_add_(0, flat, c[ok])
                pairs[1].index_add_(0, flat, one[ok])
                pairs[2].index_add_(0, flat, c[ok] * c[ok])
        self.gene, self.gp, self.gc = ([v.view(-1, K) for v in t]
                                       for t in (gene, gp, gc))
        self.gene_z = self.gene[3].sum(1) / self.gene[2].sum(1).clamp(min=1)

    def _other_folds(self, table, idx, k):
        """Sum of ``table`` at ``idx`` over the folds other than ``k`` (all of
        them when k < 0), 0 where ``idx`` is -1. The row's own fold is left
        out of the arithmetic, not subtracted afterwards."""
        v = table[idx.clamp(min=0)]
        keep = self.folds[None, :] != k[:, None]
        total = (v * keep).sum(1)
        return torch.where(idx >= 0, total, torch.zeros_like(total))

    @torch.no_grad()
    def features(self, g, r, gp, gc, k, nb, on, raw=False):
        """(values, known) of the rows, (B, n_stats) each.

        ``k`` is the fold to leave out (-1: none), ``nb`` the training rows of
        the two neighbours (-1: none), ``on`` the visible id levels: a hidden
        gene loses its gene, project and condition means, a hidden run its
        run mean and its neighbours, as a new gene or a new run would.
        """
        gene_on, run_on, prj_on, cond_on = on
        n_r = self._other_folds(self.run[1], r, k)
        rm = self._other_folds(self.run[0], r, k) / n_r.clamp(min=1)
        n_g = self._other_folds(self.gene[2], g, k)
        m_g = self._other_folds(self.gene[0], g, k) / n_g.clamp(min=1)
        sq = self._other_folds(self.gene[1], g, k) / n_g.clamp(min=1)
        sd_g = (sq - m_g ** 2).clamp(min=0).sqrt()
        n_p = self._other_folds(self.gp[1], gp, k)
        m_p = self._other_folds(self.gp[0], gp, k) / n_p.clamp(min=1)
        n_c = self._other_folds(self.gc[1], gc, k)
        m_c = self._other_folds(self.gc[0], gc, k) / n_c.clamp(min=1)
        prj, cond = gene_on & prj_on, gene_on & cond_on
        vals = [rm, m_g, sd_g, torch.log1p(n_g), m_p, torch.log1p(n_p), m_c,
                torch.log1p(n_c)]
        known = [(n_r > 0) & run_on, (n_g > 0) & gene_on,
                 (n_g > 1) & gene_on, gene_on, (n_p > 0) & prj, prj,
                 (n_c > 0) & cond, cond]
        if self.neighbours:
            for j in nb:
                jj = j.clamp(min=0).long()
                ok = ((j >= 0) & self.vis[jj] & run_on & (n_r > 0)
                      & (self.r["fold"][jj].long() != k))
                z = self._z(jj)
                vals += [z - rm, z - self.gene_z[self.r["gene"][jj].long()]]
                known += [ok, ok]
        if self.sd:
            for table, idx, n, m, on_ in ((self.gp, gp, n_p, m_p, prj),
                                          (self.gc, gc, n_c, m_c, cond)):
                sq = self._other_folds(table[2], idx, k) / n.clamp(min=1)
                vals.append((sq - m ** 2).clamp(min=0).sqrt())
                known.append((n > 1) & on_)
        v = torch.stack(vals, 1).float()
        kn = torch.stack(known, 1)
        if not raw:
            v = ((v - self.center) / self.scale).clamp(-8, 8)
        return torch.where(kn, v, torch.zeros_like(v)), kn.float()

    def rows(self, i, on, raw=False):
        """Statistics of the training rows ``i``: a visible row leaves its
        own fold out, a holdout row reads every visible row."""
        r = self.r
        k = torch.where(self.vis[i], r["fold"][i].long(),
                        torch.full_like(i, -1))
        return self.features(r["gene"][i].long(), r["run"][i].long(),
                             r["gp"][i].long(), r["gc"][i].long(), k,
                             (r["nb_up"][i], r["nb_down"][i]), on, raw)

    def external(self, g, r, gp, gc, nb, on):
        """Statistics of rows outside the training rows (the test rows)."""
        return self.features(g, r, gp, gc, torch.full_like(g, -1), nb, on)

    @torch.no_grad()
    def fit_scale(self, i):
        """Centre and spread of each statistic on the training rows ``i``."""
        on = tuple(torch.ones(len(i), dtype=torch.bool, device=self.device)
                   for _ in range(4))
        v, kn = self.rows(i, on, raw=True)
        cnt = kn.sum(0).clamp(min=1)
        mean = (v * kn).sum(0) / cnt
        var = (((v - mean) * kn) ** 2).sum(0) / cnt
        self.center, self.scale = mean, var.sqrt().clamp(min=1e-3)

    def for_predict(self):
        """Drops what only the training rows need, once the fit is done."""
        for name in ("run", "gp", "gc", "nb_up", "nb_down"):
            self.r.pop(name, None)


# --- layers, each with its 2017 alternative --------------------------------

class _Norm(nn.Module):
    """LayerNorm (2017), RMSNorm or DyT over the last dimension."""

    def __init__(self, kind, dim, eps=1e-5):
        super().__init__()
        self.kind, self.eps = kind, eps
        self.weight = nn.Parameter(torch.ones(dim))
        if kind == "layernorm":
            self.bias = nn.Parameter(torch.zeros(dim))
        elif kind == "dyt":
            self.bias = nn.Parameter(torch.zeros(dim))
            self.alpha = nn.Parameter(torch.tensor(0.5))
        elif kind != "rmsnorm":
            raise ValueError(f"unknown norm {kind}")

    def forward(self, x):
        if self.kind == "layernorm":
            return F.layer_norm(x, x.shape[-1:], self.weight, self.bias,
                                self.eps)
        if self.kind == "rmsnorm":
            dtype = x.dtype
            x = x.float()
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return (x * self.weight).to(dtype)
        return self.weight * torch.tanh(self.alpha * x) + self.bias


def _repeat_kv(x, n_rep):
    """(B, Hkv, T, D) to (B, Hkv * n_rep, T, D), each KV head shared."""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class _Attention(nn.Module):
    """Bidirectional self-attention over the tokens of a row.

    MHA (2017) has one KV head per query head, GQA one per group, MQA a single
    one. The naive path writes softmax(QK^T / sqrt(d)) V, the SDPA path calls
    the fused kernel. Softcapping needs the scores, so it forces the naive
    path.
    """

    def __init__(self, cfg):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        if d % h:
            raise ValueError("d_model must be a multiple of n_heads")
        self.h, self.hd = h, d // h
        self.kv = {"mha": h, "gqa": cfg.n_kv_heads, "mqa": 1}[cfg.attn]
        if h % self.kv:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        self.q = nn.Linear(d, h * self.hd)
        self.k = nn.Linear(d, self.kv * self.hd)
        self.v = nn.Linear(d, self.kv * self.hd)
        self.out = nn.Linear(h * self.hd, d)
        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = _Norm("rmsnorm", self.hd)
            self.k_norm = _Norm("rmsnorm", self.hd)
        self.softcap = cfg.softcap
        self.sdpa = cfg.sdpa and not cfg.softcap
        self.dropout = cfg.dropout

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv, self.hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        k = _repeat_kv(k, self.h // self.kv)
        v = _repeat_kv(v, self.h // self.kv)
        p = self.dropout if self.training else 0.0
        if self.sdpa:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            if self.softcap:
                scores = self.softcap * torch.tanh(scores / self.softcap)
            att = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            y = F.dropout(att, p) @ v
        return self.out(y.transpose(1, 2).reshape(b, t, self.h * self.hd))


class _FFN(nn.Module):
    """Two-layer MLP (2017, ReLU or GELU) or gated (SwiGLU, GeGLU).

    The gated hidden width is 2/3 of the plain one, so both have the same
    number of parameters.
    """

    def __init__(self, kind, d, mult):
        super().__init__()
        self.kind = kind
        if kind in ("relu", "gelu"):
            hidden = int(d * mult)
            self.up = nn.Linear(d, hidden)
        elif kind in ("swiglu", "geglu"):
            hidden = int(round(d * mult * 2 / 3))
            self.up = nn.Linear(d, 2 * hidden)
        else:
            raise ValueError(f"unknown ffn {kind}")
        self.down = nn.Linear(hidden, d)

    def forward(self, x):
        h = self.up(x)
        if self.kind == "relu":
            h = F.relu(h)
        elif self.kind == "gelu":
            h = F.gelu(h)
        else:
            a, g = h.chunk(2, dim=-1)
            h = a * (F.silu(g) if self.kind == "swiglu" else F.gelu(g))
        return self.down(h)


class _MoE(nn.Module):
    """Sparse mixture of FFN experts, top-k softmax routing.

    Each expert is an ``_FFN`` of 1/top_k of the dense width, so the active
    parameters per token match the dense layer. The auxiliary loss of Switch
    Transformer (n * sum of load x mean probability) keeps the load balanced.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n, self.k = cfg.moe_experts, cfg.moe_top_k
        self.router = nn.Linear(cfg.d_model, self.n, bias=False)
        self.experts = nn.ModuleList(
            _FFN(cfg.ffn, cfg.d_model, cfg.ffn_mult / self.k)
            for _ in range(self.n))
        self.aux = torch.zeros(())

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        prob = torch.softmax(self.router(x).float(), dim=-1)
        weight, chosen = prob.topk(self.k, dim=-1)
        weight = (weight / weight.sum(-1, keepdim=True)).to(x.dtype)
        out = torch.zeros_like(x)
        load = torch.zeros(self.n, device=x.device)
        for e, expert in enumerate(self.experts):
            rows, slot = (chosen == e).nonzero(as_tuple=True)
            load[e] = rows.numel()
            if rows.numel():
                y = expert(x[rows]) * weight[rows, slot, None]
                out.index_add_(0, rows, y.to(out.dtype))
        load = load / max(x.shape[0] * self.k, 1)
        self.aux = self.n * (load * prob.mean(0)).sum()
        return out.reshape(shape)


class _Block(nn.Module):
    """Attention then FFN, each in a residual with its norm.

    post (2017): x = norm(x + f(x)). pre: x = x + f(norm(x)).
    sandwich: x = x + norm(f(norm(x))).
    """

    def __init__(self, cfg):
        super().__init__()
        self.place = cfg.norm_place
        self.attn = _Attention(cfg)
        self.ffn = _MoE(cfg) if cfg.moe_experts else _FFN(
            cfg.ffn, cfg.d_model, cfg.ffn_mult)
        self.norm1 = _Norm(cfg.norm, cfg.d_model)
        self.norm2 = _Norm(cfg.norm, cfg.d_model)
        if self.place == "sandwich":
            self.post1 = _Norm(cfg.norm, cfg.d_model)
            self.post2 = _Norm(cfg.norm, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def _sub(self, x, f, norm, post):
        if self.place == "post":
            return norm(x + self.drop(f(x)))
        if self.place == "pre":
            return x + self.drop(f(norm(x)))
        return x + self.drop(post(f(norm(x))))

    def forward(self, x):
        x = self._sub(x, self.attn, self.norm1, getattr(self, "post1", None))
        return self._sub(x, self.ffn, self.norm2, getattr(self, "post2", None))


class _MLPBlock(nn.Module):
    """x + FFN(norm(x)), the attention-free residual block."""

    def __init__(self, cfg, width):
        super().__init__()
        self.norm = _Norm(cfg.norm, width)
        # half the FFN expansion of the attention blocks: the mlp mixer then
        # costs about as much per row as two attention blocks on 21 tokens
        self.ffn = _FFN(cfg.ffn, width, 2.0)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return x + self.drop(self.ffn(self.norm(x)))


class _MLPMixer(nn.Module):
    """No attention: the tokens of a row flattened into one residual MLP.

    The same tokens and the same building blocks (norm, FFN) as the
    transformer, so the comparison measures the attention alone.
    """

    def __init__(self, cfg, n_tokens):
        super().__init__()
        width = cfg.mlp_hidden
        self.inp = nn.Linear(n_tokens * cfg.d_model, width)
        self.blocks = nn.ModuleList(_MLPBlock(cfg, width)
                                    for _ in range(cfg.n_layers))
        self.norm = _Norm(cfg.norm, width)
        self.out = nn.Linear(width, cfg.d_model)

    def forward(self, x):
        x = self.inp(x.flatten(1))
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))


class _PLR(nn.Module):
    """Periodic embeddings of numbers (PLR, Gorishniy et al., 2022).

    Each number x gives sin and cos of 2 pi c x for learned frequencies c,
    then a linear layer of its own and a ReLU. A plain z * w + b is one
    direction per number, PLR lets a number act differently in each range.
    """

    def __init__(self, n, out, n_freq=8, sigma=1.0):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n, n_freq) * sigma)
        self.weight = nn.Parameter(torch.randn(n, 2 * n_freq, out)
                                   / math.sqrt(2 * n_freq))
        self.bias = nn.Parameter(torch.zeros(n, out))

    def forward(self, x):
        """(B, n) numbers to (B, n, out) embeddings."""
        v = 2 * math.pi * x.float()[..., None] * self.freq
        v = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
        return torch.relu(torch.einsum("bnk,nko->bno", v, self.weight)
                          + self.bias)


class _Numbers(nn.Module):
    """One token per number of the run sheet, or a missing token.

    z * w + b by default, a PLR embedding when ``plr_freq`` is set.
    """

    def __init__(self, n, d, plr_freq=0):
        super().__init__()
        if plr_freq:
            self.plr = _PLR(n, d, plr_freq)
        else:
            self.w = nn.Parameter(torch.randn(n, d) * 0.02)
            self.b = nn.Parameter(torch.zeros(n, d))
        self.missing = nn.Parameter(torch.randn(n, d) * 0.02)

    def forward(self, z, known):
        known = known[..., None]
        value = (self.plr(z) if hasattr(self, "plr")
                 else z[..., None] * self.w + self.b)
        return known * value + (1 - known) * self.missing


class _Slots(nn.Module):
    """Prepends the CLS token, adds one learned embedding per token slot.

    The tokens are a set, not a sequence: the slot says which field a token
    carries, the job a position embedding does in a sentence.
    """

    def __init__(self, n_tokens, d):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(d) * 0.02)
        self.slot = nn.Parameter(torch.randn(n_tokens, d) * 0.02)

    def forward(self, x):
        cls = self.cls.to(x.dtype).expand(x.shape[0], 1, -1)
        return torch.cat([cls, x], dim=1) + self.slot.to(x.dtype)


class _Biases(nn.Module):
    """Per-level additive effects (id_bias), read with index_select."""

    def __init__(self, init):
        super().__init__()
        for k, v in init.items():
            setattr(self, k, nn.Parameter(torch.as_tensor(
                v, dtype=torch.float32)))

    def forward(self, g, r, G, p, gene_on, run_on, prj_on):
        eta = (self.b0 + torch.index_select(self.a_genome, 0, G)
               + torch.index_select(self.a_gene, 0, g) * gene_on
               + torch.index_select(self.b_run, 0, r) * run_on
               + torch.index_select(self.b_project, 0, p) * prj_on)
        log_r = self.rho0 + torch.index_select(self.rho_gene, 0, g) * gene_on
        return eta, log_r


class _ExprTransformer(nn.Module):
    """Tokens of a (gene, run) row, the transformer, the head, the id biases.

    The per gene and per run tables are buffers, so a batch is only the gene
    id, the run id and which of their levels are visible.
    """

    SLOTS = ["cls", "gene", "features", "genome", "run", "project",
             "condition"]

    def __init__(self, cfg, tables, bias_init):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        t = tables
        self.n_gene, self.n_run = t["n_gene"], t["n_run"]
        self.n_prj, self.n_cond = t["n_prj"], t["n_cond"]
        for name in ("x_gene", "genome_of_gene", "prj_of_run", "cond_of_run",
                     "run_cat", "run_num", "run_known"):
            self.register_buffer(name, torch.as_tensor(t[name]),
                                 persistent=False)
        n_fields = t["run_cat"].shape[1]
        n_nums = t["run_num"].shape[1]
        # id embeddings, the last slots are MASK (unseen) and NONE (no sheet)
        self.gene_emb = nn.Embedding(self.n_gene + 1, d)
        self.genome_emb = nn.Embedding(t["n_genome"], d)
        self.run_emb = nn.Embedding(self.n_run + 1, d)
        self.prj_emb = nn.Embedding(self.n_prj + 1, d)
        self.cond_emb = nn.Embedding(self.n_cond + 2, d)
        self.field_emb = nn.Embedding(t["n_vocab"], d)
        if cfg.num_embed not in ("linear", "plr"):
            raise ValueError(f"unknown num_embed {cfg.num_embed}")
        plr = cfg.num_embed == "plr"
        # plr: each gene signal gets its own small embedding, the ESM block
        # and the flags stay linear
        self.n_signals = t.get("n_signals", 0) if plr else 0
        n_feat = t["x_gene"].shape[1]
        if self.n_signals:
            self.signal_plr = _PLR(self.n_signals, cfg.plr_dim, cfg.plr_freq)
            n_feat += self.n_signals * (cfg.plr_dim - 1)
        self.features = nn.Linear(n_feat, d)
        self.numbers = _Numbers(n_nums, d, cfg.plr_freq if plr else 0)
        self.n_tokens = len(self.SLOTS) + n_fields + n_nums
        # the frozen additive GLM (glm_eta), filled by set_glm
        self.glm_mode = cfg.glm_eta
        if self.glm_mode not in ("off", "offset", "token"):
            raise ValueError(f"unknown glm_eta {self.glm_mode}")
        if self.glm_mode != "off":
            sizes = {"b0": (), "a_genome": (t["n_genome"],),
                     "a_gene": (self.n_gene,), "b_project": (self.n_prj,),
                     "b_condition": (self.n_cond + 1,),
                     "b_run": (self.n_run,), "rho0": (),
                     "rho_genome": (t["n_genome"],),
                     "rho_gene": (self.n_gene,), "center": (4,)}
            for k, shape in sizes.items():
                self.register_buffer(f"glm_{k}", torch.zeros(shape),
                                     persistent=False)
        if self.glm_mode == "token":
            # two numbers: the GLM log mu and log sigma of the row
            self.glm_tokens = _Numbers(2, d)
            self.n_tokens += 2
        # the statistics of the training counts, one token each, the
        # missing token when a statistic does not exist for the row
        if cfg.count_stats not in ("off", "means", "all"):
            raise ValueError(f"unknown count_stats {cfg.count_stats}")
        self.n_stats = len(_stat_names(cfg.count_stats, cfg.stats_sd))
        if self.n_stats:
            self.stat_tokens = _Numbers(self.n_stats, d)
            self.n_tokens += self.n_stats
        self.slots = _Slots(self.n_tokens, d)
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.mixer not in ("attention", "mlp"):
            raise ValueError(f"unknown mixer {cfg.mixer}")
        if cfg.mixer == "mlp":
            self.mixer = _MLPMixer(cfg, self.n_tokens)
            self.blocks = nn.ModuleList()
        else:
            self.blocks = nn.ModuleList(_Block(cfg)
                                        for _ in range(cfg.n_layers))
        self.final_norm = (_Norm(cfg.norm, d) if cfg.norm_place != "post"
                           else nn.Identity())
        self.head = nn.Linear(d, 2)
        # multiplies the head outputs (log mu, log sigma), see out_scale
        self.register_buffer("head_scale", torch.ones(2), persistent=False)
        self.bias = _Biases(bias_init)
        self._init_weights()

    def _init_weights(self):
        for emb in (self.gene_emb, self.genome_emb, self.run_emb,
                    self.prj_emb, self.cond_emb, self.field_emb):
            nn.init.normal_(emb.weight, std=0.02)
        if self.cfg.init == "gpt":
            scale = 0.02 / math.sqrt(2 * self.cfg.n_layers)
            for name, module in self.blocks.named_modules():
                if isinstance(module, nn.Linear):
                    residual = name.endswith(("out", "down"))
                    nn.init.normal_(module.weight,
                                    std=scale if residual else 0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        # the head starts at 0: the first step predicts the id biases alone
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @torch.no_grad()
    def set_glm(self, glm, center):
        """Copies the GLM effects (tensors) and the token centering in."""
        for k, v in glm.items():
            getattr(self, f"glm_{k}").copy_(v)
        self.glm_center.copy_(torch.as_tensor(center))

    def glm_terms(self, g, r, gene_on, run_on, prj_on, cond_on):
        """GLM log mu (without the offset) and log sigma of each row.

        An effect counts only when its level is visible, exactly as the id
        tokens: a masked gene loses a_gene and rho_gene, a masked run b_run.
        """
        G = self.genome_of_gene[g]
        p = self.prj_of_run[r]
        c = self.cond_of_run[r]
        ci = torch.where(c < 0, self.n_cond, c)
        f = [x.float() for x in (gene_on, run_on, prj_on, cond_on)]
        eta = (self.glm_b0 + self.glm_a_genome[G] + self.glm_a_gene[g] * f[0]
               + self.glm_b_run[r] * f[1] + self.glm_b_project[p] * f[2]
               + self.glm_b_condition[ci] * f[3])
        log_r = (self.glm_rho0 + self.glm_rho_genome[G]
                 + self.glm_rho_gene[g] * f[0])
        return eta, log_r

    def _gene_inputs(self, g):
        """The gene feature vector, its signals through PLR when asked."""
        x = self.x_gene[g].to(self.features.weight.dtype)
        if not self.n_signals:
            return x
        k = self.n_signals
        sig = self.signal_plr(x[:, :k]).flatten(1)
        return torch.cat([sig, x[:, k:].to(sig.dtype)], dim=1)

    def tokens(self, g, r, gene_on, run_on, prj_on, cond_on, glm=None,
               stats=None):
        p = self.prj_of_run[r]
        c = self.cond_of_run[r]
        c_tok = torch.where(c < 0, self.n_cond + 1,
                            torch.where(cond_on, c, self.n_cond))
        nums = self.numbers(self.run_num[r], self.run_known[r])
        parts = [
            self.gene_emb(torch.where(gene_on, g, self.n_gene))[:, None],
            self.features(self._gene_inputs(g))[:, None],
            self.genome_emb(self.genome_of_gene[g])[:, None],
            self.run_emb(torch.where(run_on, r, self.n_run))[:, None],
            self.prj_emb(torch.where(prj_on, p, self.n_prj))[:, None],
            self.cond_emb(c_tok)[:, None],
            self.field_emb(self.run_cat[r]),
            nums,
        ]
        if self.glm_mode == "token":
            c0 = self.glm_center
            z = torch.stack([(glm[0] - c0[0]) / c0[1],
                             (glm[1] - c0[2]) / c0[3]], dim=1)
            parts.append(self.glm_tokens(z, torch.ones_like(z)))
        if self.n_stats:
            if stats is None:
                # no statistics given: every one reads its missing token
                z = torch.zeros(len(g), self.n_stats, device=g.device)
                stats = (z, z)
            parts.append(self.stat_tokens(*stats))
        x = torch.cat([part.to(nums.dtype) for part in parts], dim=1)
        return self.drop(self.slots(x))

    def biases(self, g, r, gene_on, run_on, prj_on):
        return self.bias(g, r, self.genome_of_gene[g], self.prj_of_run[r],
                         gene_on, run_on, prj_on)

    def forward(self, g, r, gene_on, run_on, prj_on, cond_on, stats=None):
        """log mu without the offset, log sigma, auxiliary loss.

        ``stats`` is the (values, known) pair of _CountStats (count_stats).
        """
        glm = (self.glm_terms(g, r, gene_on, run_on, prj_on, cond_on)
               if self.glm_mode != "off" else None)
        x = self.tokens(g, r, gene_on, run_on, prj_on, cond_on, glm, stats)
        if self.cfg.mixer == "mlp":
            h = self.mixer(x)
        else:
            for block in self.blocks:
                x = block(x)
            h = x[:, 0]
        out = self.head(self.final_norm(h)).float() * self.head_scale
        eta, log_r = out[:, 0], out[:, 1]
        if self.cfg.id_bias:
            be, br = self.biases(g, r, gene_on.float(), run_on.float(),
                                 prj_on.float())
            eta, log_r = eta + be, log_r + br
        else:
            eta = eta + self.bias.b0
            log_r = log_r + self.bias.rho0
        if self.glm_mode == "offset":
            # the network learns a correction to the GLM prediction
            eta, log_r = eta + glm[0], log_r + glm[1]
        aux = sum((blk.ffn.aux for blk in self.blocks
                   if isinstance(blk.ffn, _MoE)), torch.zeros((), device=g.device))
        return eta, log_r, aux


# --- the model ---------------------------------------------------------------

def _schedule_step(step, total, elapsed, budget):
    """Step the schedule reads: the real one, or the share of the time spent.

    Equal to ``step`` when the run ends on its step count. Ahead of it when
    the clock would stop the run first.
    """
    if budget <= 0:
        return step
    by_time = int(total * min(elapsed / budget, 1.0))
    return min(max(step, by_time), total - 1)


def _lr_factor(cfg, step, total):
    """Learning rate multiplier at ``step`` of ``total``."""
    warm = max(cfg.warmup_steps, 1)
    if cfg.schedule == "noam":
        # 2017: linear warmup, then 1 / sqrt(step), peak 1 at the warmup
        s = step + 1
        return min(s / warm, math.sqrt(warm / s))
    if step < warm:
        return (step + 1) / warm
    if cfg.schedule == "constant":
        return 1.0
    progress = (step - warm) / max(total - 1 - warm, 1)
    if cfg.schedule == "cosine":
        return 0.1 + 0.45 * (1 + math.cos(math.pi * min(progress, 1.0)))
    if cfg.schedule == "wsd":
        # constant plateau, linear decay over the last 20 % of the steps
        return 1.0 if progress < 0.8 else 1.0 - 0.9 * (progress - 0.8) / 0.2
    raise ValueError(f"unknown schedule {cfg.schedule}")


class TransformerModel:
    """Fit the expression transformer and predict ``(mu, sigma)``."""

    def __init__(self, meta, cfg=None, device="cpu"):
        self.genes = meta["feature_annotations"]
        self.submission_dir = meta.get("submission_dir")
        self.cfg = cfg or TConfig()
        self.device = device
        # Hooks the local harness replaces with RunLogger methods
        self.log = _print_log
        self.metric = _no_metric
        self.track = lambda iterable, desc="train", total=None: iterable
        # factory(module) -> LayerStats, set by the harness to log layer/...
        self.layer_stats = None
        self.net = None
        # _CountStats of the fit (count_stats), None when off
        self.stats = None

    # --- data ---------------------------------------------------------

    def prepare(self, train_df, samples_df):
        """Index, pairs, per gene and per run tables, before any training.

        Returns the gene id, run id, count and offset of each training row.
        Also used by visualisation/build_arch.py to build the real model.
        With ``BACTERIA_KEEP_PREPARED=1`` (tools/ablate.py) the tables are
        kept for the next fit on the same rows. The platform fits once.
        """
        key = (id(train_df), len(train_df), id(samples_df))
        self._prep_key = key
        if key in _PREPARED:
            (self.ix, self.gp, self.gc, self.tables, self.family,
             self.n_family, out) = _PREPARED[key]
            self.log("prepared tables of the previous fit reused")
            return out
        out = self._prepare(train_df, samples_df)
        if os.environ.get(KEEP_ENV) == "1":
            _PREPARED.clear()
            _PREPARED[key] = (self.ix, self.gp, self.gc, self.tables,
                              self.family, self.n_family, out)
        return out

    def _prepare(self, train_df, samples_df):
        t0 = time.time()
        self.ix = ix = _Index(self.genes, samples_df)
        target = "readcount" if "readcount" in train_df else "counts"
        gene, _, run = ix.rows(train_df)
        ok = (gene >= 0) & (run >= 0)
        y = train_df[target].to_numpy(dtype=np.float32)
        off = train_df["log_length_ratio"].to_numpy(dtype=np.float32) \
            + train_df["log_size_factor"].to_numpy(dtype=np.float32)
        if not ok.all():
            self.log(f"dropping {int((~ok).sum())} rows of unknown gene/run")
            gene, run, y, off = (a[ok] for a in (gene, run, y, off))
        self.gp = _Pairs(gene, ix.prj_of_run[run], ix.n_prj)
        self.gc = _Pairs(gene, ix.cond_of_run[run], ix.n_cond)
        _gc.collect()
        cat, n_vocab, num, known = _run_tokens(samples_df)
        x_gene, n_signals = _gene_features(self.genes, self.submission_dir,
                                           self.log)
        self.tables = {
            "n_gene": ix.n_gene, "n_run": ix.n_run, "n_prj": ix.n_prj,
            "n_cond": ix.n_cond, "n_genome": ix.n_genome, "n_vocab": n_vocab,
            "x_gene": x_gene, "n_signals": n_signals,
            "genome_of_gene": ix.genome_of_gene.astype(np.int64),
            "prj_of_run": ix.prj_of_run.astype(np.int64),
            "cond_of_run": ix.cond_of_run.astype(np.int64),
            "run_cat": cat, "run_num": num, "run_known": known,
        }
        self.family = _gene_families(self.genes)
        self.n_family = int(self.family.max()) + 1
        self.log(f"{len(y):,} rows, {ix.n_gene:,} genes, {ix.n_run:,} runs, "
                 f"{ix.n_prj:,} projects, {ix.n_cond:,} conditions, "
                 f"{cat.shape[1]} run fields ({n_vocab} values), "
                 f"{self.tables['x_gene'].shape[1]} gene inputs "
                 f"({time.time() - t0:.0f}s)")
        return gene, run, y, off

    def _moments(self, gene, run, y, off, rows):
        """Moment estimates of b0, a_gene and b_run on the log scale."""
        ix = self.ix
        z = np.log(y[rows] + 0.5) - off[rows]
        g, r = gene[rows], run[rows]
        b0 = float(z.mean())
        a_gene = (np.bincount(g, z - b0, ix.n_gene)
                  / (np.bincount(g, minlength=ix.n_gene) + 1.0))
        res = z - b0 - a_gene[g]
        b_run = (np.bincount(r, res, ix.n_run)
                 / (np.bincount(r, minlength=ix.n_run) + 1.0))
        return {"b0": b0, "a_gene": a_gene, "b_run": b_run}

    def _bias_init(self, gene, run, y, off, rows):
        """Moment estimates on the log scale, the per-level warm start."""
        ix = self.ix
        m = self._moments(gene, run, y, off, rows)
        b0, a_gene, b_run = m["b0"], m["a_gene"], m["b_run"]
        zero = np.zeros
        if self.cfg.glm_eta == "offset":
            # the GLM already carries the level: the network starts at 0
            b0, a_gene, b_run = 0.0, zero(ix.n_gene), zero(ix.n_run)
            return {"b0": np.array(0.0), "a_genome": zero(ix.n_genome),
                    "a_gene": a_gene, "b_run": b_run,
                    "b_project": zero(ix.n_prj), "rho0": np.array(0.0),
                    "rho_gene": zero(ix.n_gene)}
        return {"b0": np.array(b0), "a_genome": zero(ix.n_genome),
                "a_gene": a_gene, "b_run": b_run,
                "b_project": zero(ix.n_prj), "rho0": np.array(np.log(5.0)),
                "rho_gene": zero(ix.n_gene)}

    def _fit_glm(self, gene, run, y, off, rows, steps, stage, init=None):
        """Additive NB GLM on ``rows``, full batch Adam over GPU chunks.

        The reference GLM without its interactions: genome, gene, project,
        condition and run effects, each pulled to 0 by its L2 prior. Returns
        the effects as device tensors, frozen afterwards.
        """
        dev, t, ix = self.device, self.t, self.ix
        t_start = time.time()
        sel = torch.as_tensor(np.flatnonzero(rows)).to(dev)
        n = len(sel)
        if init is None:
            b = self._moments(gene, run, y, off, rows)
            zero = np.zeros
            init = {"b0": np.array(b["b0"]), "a_genome": zero(ix.n_genome),
                    "a_gene": b["a_gene"], "b_project": zero(ix.n_prj),
                    "b_condition": zero(ix.n_cond + 1), "b_run": b["b_run"],
                    "rho0": np.array(np.log(5.0)),
                    "rho_genome": zero(ix.n_genome),
                    "rho_gene": zero(ix.n_gene)}
        P = {k: torch.tensor(np.asarray(v.cpu() if torch.is_tensor(v) else v),
                             dtype=torch.float32, device=dev,
                             requires_grad=True) for k, v in init.items()}
        G_of = torch.as_tensor(ix.genome_of_gene).long().to(dev)
        p_of = torch.as_tensor(ix.prj_of_run).long().to(dev)
        c_of = torch.as_tensor(np.where(ix.cond_of_run >= 0, ix.cond_of_run,
                                        ix.n_cond)).long().to(dev)
        opt = torch.optim.Adam(list(P.values()), lr=0.05)
        first = last = math.nan
        for step in range(steps):
            for group in opt.param_groups:
                group["lr"] = 0.05 * (0.1 + 0.45 * (1 + math.cos(
                    math.pi * step / steps)))
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for s in range(0, n, GPU_CHUNK):
                i = sel[s:s + GPU_CHUNK]
                g, r = t["gene"][i].long(), t["run"][i].long()
                G, pr, c = G_of[g], p_of[r], c_of[r]

                def take(k, idx):
                    # index_select: the backward of P[k][idx] is 26 times
                    # slower on rows sorted by genome and run
                    return torch.index_select(P[k], 0, idx)

                eta = (t["off"][i] + P["b0"] + take("a_genome", G)
                       + take("a_gene", g) + take("b_project", pr)
                       + take("b_condition", c) + take("b_run", r))
                log_r = (P["rho0"] + take("rho_genome", G)
                         + take("rho_gene", g))
                loss = _nb_nll_torch(t["y"][i], eta, log_r).sum() / n
                loss.backward()
                total += float(loss.detach())
            penalty = sum((P[k] ** 2).sum() / (2 * sd ** 2)
                          for k, sd in GLM_PRIOR_SD.items()) / n
            penalty.backward()
            opt.step()
            with torch.no_grad():
                P["b_condition"][-1] = 0.0
            first = total if step == 0 else first
            last = total
        self.log(f"{stage}: {n:,} rows, {steps} steps, NLL without lgamma(y+1)"
                 f" {first:.4f} to {last:.4f} ({time.time() - t_start:.0f}s)")
        return {k: v.detach() for k, v in P.items()}

    def _glm_center(self, glm, idx):
        """Mean and spread of the GLM terms on some rows, for the tokens."""
        net, t = self.net, self.t
        g, r = t["gene"][idx].long(), t["run"][idx].long()
        on = self._on(g, r, {k: torch.ones(n, dtype=torch.bool,
                                           device=self.device)
                             for k, n in (("gene", net.n_gene),
                                          ("run", net.n_run),
                                          ("prj", net.n_prj),
                                          ("cond", net.n_cond))})
        eta, log_r = net.glm_terms(g, r, *on)
        return [float(eta.mean()), float(eta.std().clamp(min=1e-3)),
                float(log_r.mean()), float(log_r.std().clamp(min=1e-3))]

    def _count_stats(self, gene, run, fit_rows):
        """The _CountStats of the training rows, its tables on the fit rows.

        The neighbour rows (the training row of each genome neighbour in the
        same run) are found once and kept like the prepared tables.
        """
        cfg, dev, ix, t = self.cfg, self.device, self.ix, self.t
        t_start = time.time()
        key = getattr(self, "_prep_key", None)
        if key is not None and key in _NEIGHBOURS:
            cached = _NEIGHBOURS[key]
        else:
            up_gene, down_gene, _, _ = _neighbours(self.genes)
            row_key, row_order = _row_lookup(gene, run, ix.n_run)
            cached = (up_gene, down_gene, row_key, row_order,
                      _find_rows(row_key, row_order, up_gene, gene, run,
                                 ix.n_run),
                      _find_rows(row_key, row_order, down_gene, gene, run,
                                 ix.n_run))
            if os.environ.get(KEEP_ENV) == "1":
                _NEIGHBOURS.clear()
                _NEIGHBOURS[key] = cached
        (self.up_gene, self.down_gene, self.row_key, self.row_order,
         up, down) = cached
        fold = _hash_fold(gene, _gene_blocks(gene, run, ix,
                                             cfg.project_share),
                          cfg.stats_folds)
        rows = {"gene": t["gene"], "run": t["run"], "y": t["y"],
                "off": t["off"],
                "gp": torch.as_tensor(self.gp.idx).to(dev),
                "gc": torch.as_tensor(self.gc.idx).to(dev),
                "fold": torch.as_tensor(fold).to(dev),
                "nb_up": torch.as_tensor(up).to(dev),
                "nb_down": torch.as_tensor(down).to(dev)}
        stats = _CountStats(rows, ix.n_gene, ix.n_run, self.gp.n, self.gc.n,
                            cfg.stats_folds, cfg.count_stats == "all", dev,
                            sd=cfg.stats_sd)
        stats.build(torch.as_tensor(fit_rows).to(dev))
        some = np.flatnonzero(fit_rows)
        some = some[::max(1, len(some) // 200_000)]
        stats.fit_scale(torch.as_tensor(some).to(dev))
        found = float(np.mean((up >= 0) | (down >= 0)))
        self.log(f"count stats: {stats.n_stats} inputs, {cfg.stats_folds} "
                 f"folds of gene x block, a neighbour row for {found:.0%} of "
                 f"the rows ({time.time() - t_start:.0f}s), "
                 + (f"GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
                    if dev != "cpu" else "") + _ram())
        return stats

    def _stat_inputs(self, i, on):
        """Count statistics of the training rows ``i``, None when off."""
        return self.stats.rows(i, on) if self.stats is not None else None

    def _seen(self, gene, run, rows):
        """Levels seen in ``rows``, as numpy and as device masks."""
        ix = self.ix
        g, r = gene[rows], run[rows]
        cond = ix.cond_of_run[r]
        gp, gc = self.gp.idx[rows], self.gc.idx[rows]
        seen = {
            "gene": np.bincount(g, minlength=ix.n_gene) > 0,
            "run": np.bincount(r, minlength=ix.n_run) > 0,
            "prj": np.bincount(ix.prj_of_run[r], minlength=ix.n_prj) > 0,
            "cond": np.bincount(cond[cond >= 0], minlength=ix.n_cond) > 0,
            "gp": np.bincount(gp, minlength=max(self.gp.n, 1)) >= 2,
            "gc": np.bincount(gc[gc >= 0], minlength=max(self.gc.n, 1)) >= 2,
        }
        dev = {k: torch.as_tensor(seen[k]).to(self.device)
               for k in ("gene", "run", "prj", "cond")}
        return seen, dev

    def _on(self, g, r, seen_dev):
        """Which id tokens of each row the model may read."""
        p = self.net.prj_of_run[r]
        c = self.net.cond_of_run[r]
        return (seen_dev["gene"][g], seen_dev["run"][r], seen_dev["prj"][p],
                seen_dev["cond"][c.clamp(min=0)] & (c >= 0))

    def _case(self, seen, gene, run, gp, gc):
        """Case of each row, the index into CASES (numpy)."""
        ix = self.ix
        g_ok, r_ok = gene >= 0, run >= 0
        gi, ri = np.where(g_ok, gene, 0), np.where(r_ok, run, 0)
        p, c = ix.prj_of_run[ri], ix.cond_of_run[ri]
        sg = g_ok & seen["gene"][gi]
        sr = r_ok & seen["run"][ri]
        sp = r_ok & seen["prj"][p]
        sc = r_ok & (c >= 0) & seen["cond"][np.maximum(c, 0)]
        vgp = (gp >= 0) & seen["gp"][np.maximum(gp, 0)] & sg
        vgc = (gc >= 0) & seen["gc"][np.maximum(gc, 0)] & sg
        return np.select([sg & sr & (vgp | vgc), sg & sr, sg & sc, sg & sp,
                          sg, sr], [0, 1, 2, 3, 4, 5], 6)

    # --- training -----------------------------------------------------

    def _optimizer(self, net):
        cfg = self.cfg
        if cfg.optimizer == "adam":
            return torch.optim.Adam(net.parameters(), lr=cfg.lr,
                                    betas=(0.9, 0.98), eps=1e-9)
        if cfg.optimizer != "adamw":
            raise ValueError(f"unknown optimizer {cfg.optimizer}")
        decay, rest = [], []
        for name, p in net.named_parameters():
            # weight decay on the matrices of the blocks and the projections
            # only, never on norms, biases, embeddings or the per-level
            # effects
            matrix = p.dim() == 2 and not name.startswith(
                ("gene_emb", "genome_emb", "run_emb", "prj_emb", "cond_emb",
                 "field_emb", "slots.", "numbers.", "stat_tokens.", "bias."))
            (decay if matrix else rest).append(p)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": rest, "weight_decay": 0.0}],
            lr=cfg.lr, betas=(0.9, 0.95))

    def _amp(self):
        cfg = self.cfg
        on = self.device != "cpu" and cfg.precision in ("fp16", "bf16")
        dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda" if on else "cpu",
                              dtype=dtype, enabled=on)

    def _cold(self, step):
        """Level dropout of one step: the levels whose id tokens are hidden."""
        cfg, dev = self.cfg, self.device
        gen = torch.Generator(device=dev)
        gen.manual_seed(getattr(self, "_seed", cfg.seed) * 100_003 + step)

        def draw(n, p):
            return torch.rand(n, generator=gen, device=dev) < p

        net = self.net
        cold_gene = draw(self.n_family, cfg.drop_gene)[self.family_dev]
        cold_cond = draw(net.n_cond, cfg.drop_cond)
        cold_prj = draw(net.n_prj, cfg.drop_prj)
        c = net.cond_of_run
        cold_run = (draw(net.n_run, cfg.drop_run)
                    | (cold_cond[c.clamp(min=0)] & (c >= 0))
                    | cold_prj[net.prj_of_run])
        return cold_gene, cold_run, cold_prj, cold_cond

    @torch.no_grad()
    def _forward_rows(self, idx, seen_dev, net=None):
        """log mu and log sigma (before the case shift) of the rows ``idx``."""
        net, t = net or self.net, self.t
        was = net.training
        net.eval()
        etas, logs = [], []
        for s in range(0, len(idx), EVAL_CHUNK):
            i = idx[s:s + EVAL_CHUNK]
            g, r = t["gene"][i].long(), t["run"][i].long()
            on = self._on(g, r, seen_dev)
            st = self._stat_inputs(i, on)
            with self._amp():
                eta, log_r, _ = net(g, r, *on, stats=st)
            etas.append(eta.float() + t["off"][i])
            logs.append(log_r.float())
        net.train(was)
        return torch.cat(etas), torch.cat(logs)

    def _curve(self, idx, seen_dev):
        eta, log_r = self._forward_rows(idx, seen_dev)
        return float(_nb_nll_capped(self.t["y"][idx], eta, log_r).mean())

    def _train(self, rows, epochs, stage, deadline, curves=None, step0=0,
               lr_scale=1.0):
        """Adam on the NB NLL of ``rows``, batch by batch.

        ``curves`` holds the fixed samples of the train and validation curves,
        drawn every ``eval_every`` steps. The best checkpoint on the
        validation mean is kept when ``keep_best``.
        """
        cfg, dev, net, t = self.cfg, self.device, self.net, self.t
        sel = torch.as_tensor(np.flatnonzero(rows)).to(dev)
        n = len(sel)
        steps = max(1, int(math.ceil(epochs * n / cfg.batch)))
        opt = self._optimizer(net)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=dev != "cpu" and cfg.precision == "fp16")
        stats = self.layer_stats(net) if self.layer_stats else None
        if stats is not None:
            self.log(stats.describe())
        self.log(f"{stage}: {n:,} rows, {steps:,} steps of {cfg.batch:,} "
                 f"({epochs:g} epoch), {_ram()}")
        best, best_state, best_step = math.inf, None, -1
        perm, pos = None, n
        t_start = time.time()
        # time spent on the curves, left out of the throughput
        t_eval = 0.0
        rows_done = done = 0
        for step in self.track(range(steps), desc=stage, total=steps):
            if time.time() > deadline:
                self.log(f"{stage}: time budget reached at step {step:,}")
                break
            if pos + cfg.batch > n:
                perm = sel[torch.randperm(n, device=dev)]
                pos = 0
            i = perm[pos:pos + cfg.batch]
            pos += cfg.batch
            # the schedule follows the steps or the clock, whichever is
            # further: when the time budget binds (a slower GPU), the decay
            # still ends at the deadline
            lr = cfg.lr * lr_scale * _lr_factor(
                cfg, _schedule_step(step, steps, time.time() - t_start,
                                    deadline - t_start), steps)
            for group in opt.param_groups:
                group["lr"] = lr
            if stats is not None:
                stats.start(step0 + step)

            g, r = t["gene"][i].long(), t["run"][i].long()
            cold_gene, cold_run, cold_prj, cold_cond = self._cold(step0 + step)
            c = net.cond_of_run[r]
            on = (~cold_gene[g], ~cold_run[r], ~cold_prj[net.prj_of_run[r]],
                  ~cold_cond[c.clamp(min=0)] & (c >= 0))
            # the level dropout hides the statistics of a cold level too
            st = self._stat_inputs(i, on)
            with self._amp():
                eta, log_r, aux = net(g, r, *on, stats=st)
            nll = _nb_nll_torch(t["y"][i], eta.float() + t["off"][i],
                                log_r.float()).mean()
            loss = nll + cfg.moe_aux * aux
            if cfg.id_bias and cfg.bias_prior > 0:
                # the GLM prior on the id biases, spread over the rows
                loss = loss + cfg.bias_prior * sum(
                    (getattr(net.bias, k) ** 2).sum() / (2 * sd ** 2)
                    for k, sd in BIAS_PRIOR_SD.items()) / n
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if stats is not None:
                stats.before_optimizer()
            grad_norm = None
            if cfg.grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(net.parameters(),
                                                     cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            rows_done += len(i)
            done += 1
            layer = stats.collect() if stats is not None else {}

            values = {}
            if step % 10 == 0 or step == steps - 1:
                # the curve of the batch loss, lgamma(y + 1) added back
                values["train/batch_nll"] = float(
                    nll.detach() + torch.lgamma(t["y"][i] + 1).mean())
                values["train/lr"] = lr
                values["throughput/rows_s"] = rows_done / max(
                    time.time() - t_start - t_eval, 1e-6)
                if grad_norm is not None:
                    values["train/grad_norm"] = float(grad_norm)
            due = curves is not None and (
                step % cfg.eval_every == 0 or step == steps - 1)
            if due:
                t_curve = time.time()
                values.update(self._curves(curves))
                t_eval += time.time() - t_curve
                score = values["val/nb_nll"]
                if score < best:
                    best, best_step = score, step0 + step
                    if cfg.keep_best:
                        best_state = {k: v.detach().clone()
                                      for k, v in net.state_dict().items()}
            if values or layer:
                self.metric(step=step0 + step, **values, **layer)
        if stats is not None:
            stats.remove()
        if curves is not None:
            final = self._curves(curves)
            if final["val/nb_nll"] < best:
                best, best_step, best_state = (final["val/nb_nll"],
                                               step0 + done, None)
            self.metric(step=step0 + done, **final)
            if best_state is not None:
                net.load_state_dict(best_state)
                self.log(f"{stage}: best checkpoint kept, step {best_step:,}, "
                         f"val {best:.4f}")
            self.best = {"best/val_nb_nll": best, "best/step": best_step}
        elapsed = time.time() - t_start
        speed = rows_done / max(elapsed - t_eval, 1e-6)
        self.log(f"{stage}: {done:,} steps, {rows_done:,} rows in "
                 f"{elapsed:.0f}s, {speed:,.0f} rows/s ({t_eval:.0f}s of "
                 f"curves)")
        self.speed = speed
        del opt, scaler, perm, sel
        return step0 + done

    def _curves(self, curves):
        out = {"train/nb_nll": self._curve(curves["train"], curves["seen"])}
        for name in ("A", "B", "C", "D"):
            if name in curves:
                out[f"val/nb_nll_{name}"] = self._curve(curves[name],
                                                        curves["seen"])
        ranked = [out[f"val/nb_nll_{k}"] for k in ("A", "B", "C")
                  if f"val/nb_nll_{k}" in out]
        out["val/nb_nll"] = float(np.mean(ranked)) if ranked else math.nan
        return out

    # --- fit / predict ------------------------------------------------

    def _holdouts(self, gene, run):
        """Fit rows and the internal holdout of each region (numpy masks).

        C: 1 gene family in 20. B: runs with a sheet whose condition (1 in
        25), experiment (1 in 50) or project (1 in 40) is held out, 60 % of
        the region B rows having a new project and a new condition. A: see
        ``a_holdout``.
        """
        cfg, ix = self.cfg, self.ix
        hold_gene = _hit(self.family, 20, seed=0)[gene]
        cond = ix.cond_of_run
        run_out = ix.has_sheet & (_hit(np.maximum(cond, 0), 25, seed=1)
                                  | _hit(ix.exp_of_run, 50, seed=2)
                                  | _hit(ix.prj_of_run, 40, seed=3))
        hold_run = run_out[run]
        free = ~hold_gene & ~hold_run
        if cfg.a_holdout == "block":
            # region A hides a gene in a whole project x condition block: no
            # replicate of a holdout row stays visible (2026-11-07), and in
            # the whole project for project_share of the pairs
            keys = _gene_blocks(gene, run, ix, cfg.project_share)
            hold_block = _hash_hit(gene, keys, 25) & free
            keys = None
            hold_pair = np.zeros_like(free)
        elif cfg.a_holdout == "pair":
            block_of_run = np.where(cond >= 0, cond, ix.n_cond
                                    + ix.prj_of_run).astype(np.int32)
            hold_block = _hash_hit(gene, block_of_run[run], 50) & free
            hold_pair = _hash_hit(gene, run, 50) & free & ~hold_block
        else:
            raise ValueError(f"unknown a_holdout {cfg.a_holdout}")
        fit_rows = ~(hold_gene | hold_run | hold_block | hold_pair)
        groups = {"A": hold_block | hold_pair, "B": hold_run & ~hold_gene,
                  "C": hold_gene & ~hold_run, "D": hold_gene & hold_run}
        return fit_rows, groups

    def fit(self, train_df, samples_df, genomes_dir):
        t0 = time.time()
        cfg, dev = self.cfg, self.device
        budget = cfg.max_minutes * 60
        torch.manual_seed(cfg.seed)
        gene, run, y, off = self.prepare(train_df, samples_df)

        # 1. Internal holdouts built like the regions C, B and A
        fit_rows, groups = self._holdouts(gene, run)
        self.log("holdout rows: " + ", ".join(
            f"{k} {int(v.sum()):,}" for k, v in groups.items())
            + f", fit {int(fit_rows.sum()):,}, {_ram()}")

        # 2. Device tables and the network
        self.t = {"gene": torch.as_tensor(gene).to(dev),
                  "run": torch.as_tensor(run).to(dev),
                  "y": torch.as_tensor(y).to(dev),
                  "off": torch.as_tensor(off).to(dev)}
        self.family_dev = torch.as_tensor(self.family).long().to(dev)
        # statistics of the counts, on the fit rows only: a holdout row
        # reads them as a test row will read those of every training row
        self.stats = (self._count_stats(gene, run, fit_rows)
                      if cfg.count_stats != "off" else None)
        bias = self._bias_init(gene, run, y, off, fit_rows)
        self.net = _ExprTransformer(cfg, self.tables, bias).to(dev)
        # out_scale: the head predicts log mu in units of the spread of
        # log(y + 0.5) - offset on the fit rows (about 1.9 on the train rows)
        some = np.flatnonzero(fit_rows)[::20]
        self.head_scale = [float(np.std(np.log(y[some] + 0.5) - off[some]))
                           if cfg.out_scale else 1.0, 1.0]
        self.net.head_scale.copy_(torch.as_tensor(self.head_scale))
        glm = center = None
        if cfg.glm_eta != "off":
            # fitted on the fit rows only: its effects on a holdout level
            # stay at 0, and the level is masked anyway
            glm = self._fit_glm(gene, run, y, off, fit_rows, cfg.glm_steps,
                                "glm fit")
            self.net.set_glm(glm, [0.0, 1.0, 0.0, 1.0])
            some = torch.as_tensor(np.flatnonzero(fit_rows)[::50]).to(dev)
            center = self._glm_center(glm, some)
            self.net.set_glm(glm, center)
        n_params = sum(p.numel() for p in self.net.parameters())
        n_blocks = sum(p.numel() for p in self.net.blocks.parameters())
        self.log(f"network: {self.net.n_tokens} tokens of {cfg.d_model}, "
                 f"{cfg.n_layers} {cfg.mixer} layers, {n_params:,} parameters "
                 f"({n_blocks:,} in the blocks), "
                 + (f"GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
                    if dev != "cpu" else "") + _ram())

        # 3. Fit with the train and validation curves
        seen, seen_dev = self._seen(gene, run, fit_rows)
        rng = np.random.default_rng(cfg.seed)

        def sample(mask):
            idx = np.flatnonzero(mask)
            if len(idx) > CURVE_ROWS:
                idx = np.sort(rng.choice(idx, CURVE_ROWS, replace=False))
            return torch.as_tensor(idx).to(dev)

        curves = {"train": sample(fit_rows), "seen": seen_dev}
        curves.update({k: sample(v) for k, v in groups.items() if v.any()})
        # n_models networks of seeds seed, seed + 1..., one after the other,
        # each with its share of the time budget
        K = max(cfg.n_models, 1)
        fit_end = t0 + 0.8 * budget
        t_fit = time.time()
        self.nets, step = [], 0
        for k in range(K):
            self._seed = cfg.seed + k
            if k:
                torch.manual_seed(self._seed)
                self.net = _ExprTransformer(cfg, self.tables, bias).to(dev)
                self.net.head_scale.copy_(torch.as_tensor(self.head_scale))
                if glm is not None:
                    self.net.set_glm(glm, center)
            step = self._train(
                fit_rows, cfg.epochs, "fit" if K == 1 else f"fit {k + 1}/{K}",
                deadline=t_fit + (k + 1) / K * (fit_end - t_fit),
                curves=curves, step0=step)
            self.nets.append(self.net)
        self.log(f"fit done ({time.time() - t0:.0f}s)")

        # 4. One dispersion shift per case, with the challenge metric, on
        #    the mixture of the networks
        val = torch.as_tensor(np.flatnonzero(~fit_rows)).to(dev)
        eta, log_r = _mix(*zip(*(self._forward_rows(val, seen_dev, net)
                                 for net in self.nets), strict=True))
        val = val.cpu().numpy()
        case = self._case(seen, gene[val], run[val], self.gp.idx[val],
                          self.gc.idx[val])
        self.delta = self._calibrate(y[val], eta.cpu().numpy(),
                                     log_r.cpu().numpy(), case)
        extra = _parse_shifts(cfg.sigma_extra)
        if extra:
            self.delta = {k: v + extra.get(k, 0.0)
                          for k, v in self.delta.items()}
            self.log("dispersion shifts with the local regions: " + ", ".join(
                f"{k} {v:+.2f}" for k, v in self.delta.items()))
        eta = log_r = None

        # 5. Refit on every row, so the holdout levels get their embeddings.
        #    They start from the MASK token: a level the refit barely reaches
        #    (0.25 epoch, or cut by the clock) is then still predicted as an
        #    unseen one, never from an untrained random vector. The GLM is
        #    refitted first, warm started, so its effects cover them too.
        self.seen_fit = seen
        all_rows = np.ones(len(y), dtype=bool)
        if glm is not None:
            glm = self._fit_glm(gene, run, y, off, all_rows,
                                cfg.glm_refit_steps, "glm refit", init=glm)
        if self.stats is not None:
            # every row visible now, each still without its own fold. The
            # centre and spread of the inputs stay those of the fit.
            self.stats.build(torch.ones(len(y), dtype=torch.bool,
                                        device=dev))
        t_ref = time.time()
        for k, net in enumerate(self.nets):
            self.net, self._seed = net, cfg.seed + k
            if glm is not None:
                net.set_glm(glm, center)
            self._start_from_mask(seen_dev)
            if cfg.refit_epochs > 0:
                step = self._train(
                    all_rows, cfg.refit_epochs,
                    "refit" if K == 1 else f"refit {k + 1}/{K}",
                    deadline=t_ref + (k + 1) / K * (t0 + budget - t_ref),
                    step0=step, lr_scale=0.3)
        self.net = self.nets[0]
        self.seen, self.seen_dev = self._seen(gene, run, all_rows)
        self.metric(**self.holdout, **getattr(self, "best", {}))
        # what predict needs: the network, the index, the pairs lookups. The
        # row ids of the pairs stay when the next fit reuses them.
        if os.environ.get(KEEP_ENV) != "1":
            self.gp.idx = self.gc.idx = None
        if self.stats is not None:
            self.stats.for_predict()
        self.t = None
        _gc.collect()
        if dev != "cpu":
            torch.cuda.empty_cache()
        self.log(f"fit total {time.time() - t0:.0f}s, {_ram()}")
        return self

    @torch.no_grad()
    def _start_from_mask(self, seen_dev):
        """Copies the MASK embedding into every level unseen in the fit."""
        net = self.net
        for emb, seen, mask in ((net.gene_emb, seen_dev["gene"], net.n_gene),
                                (net.run_emb, seen_dev["run"], net.n_run),
                                (net.prj_emb, seen_dev["prj"], net.n_prj),
                                (net.cond_emb, seen_dev["cond"], net.n_cond)):
            unseen = torch.nonzero(~seen).flatten()
            if len(unseen):
                emb.weight[unseen] = emb.weight[mask].clone()

    def _calibrate(self, y, eta, log_r, case):
        rng = np.random.default_rng(0)
        mu = np.exp(np.clip(eta, -12, 18))
        delta, scores, rows = {}, {}, {}
        for k, name in enumerate(CASES):
            idx = np.flatnonzero(case == k)
            rows[name] = len(idx)
            if len(idx) < MIN_CALIB_ROWS:
                continue
            if len(idx) > MAX_CALIB_ROWS:
                idx = rng.choice(idx, MAX_CALIB_ROWS, replace=False)
            res = minimize_scalar(
                lambda d: nb_nll(y[idx], mu[idx],
                                 np.exp(np.clip(log_r[idx] + d, -7, 12))),
                bounds=(-8.0, 8.0), method="bounded",
                options={"xatol": 1e-3})
            delta[name], scores[name] = float(res.x), float(res.fun)
        for name in CASES:
            hop, visited = name, set()
            while hop not in delta and hop in FALLBACK and hop not in visited:
                visited.add(hop)
                hop = FALLBACK[hop]
            delta.setdefault(name, delta.get(hop, 0.0))
        self.log("holdout rows: " + ", ".join(
            f"{k} {v:,}" for k, v in rows.items()))
        self.log("holdout NB-NLL: " + ", ".join(
            f"{k} {v:.3f}" for k, v in scores.items()))
        self.log("dispersion shifts: " + ", ".join(
            f"{k} {v:+.2f}" for k, v in delta.items()))
        self.holdout = {**{f"holdout/nb_nll_{k}": v
                           for k, v in scores.items()},
                        **{f"holdout/delta_{k}": v for k, v in delta.items()}}
        return delta

    @torch.no_grad()
    def predict(self, test_df, samples_df, genomes_dir):
        n = len(test_df)
        if self.net is None:
            return np.column_stack([np.ones(n), np.ones(n)])
        dev = self.device
        nets = getattr(self, "nets", None) or [self.net]
        for net in nets:
            net.eval()
        gene, _, run = self.ix.rows(test_df)
        off = (test_df["log_length_ratio"].to_numpy(dtype=np.float64)
               + test_df["log_size_factor"].to_numpy(dtype=np.float64))
        off = np.nan_to_num(off)
        bad = (gene < 0) | (run < 0)
        if bad.any():
            self.log(f"predict: {int(bad.sum())} rows of an unknown gene or "
                     f"run, read as unseen")
        gi, ri = np.maximum(gene, 0), np.maximum(run, 0)
        gp = self.gp.lookup(np.where(gene >= 0, gene, -1),
                            np.where(run >= 0, self.ix.prj_of_run[ri], -1))
        gc = self.gc.lookup(np.where(gene >= 0, gene, -1),
                            np.where(run >= 0, self.ix.cond_of_run[ri], -1))
        case = self._case(self.seen, gene, run, gp, gc)
        shifts = np.array([self.delta[name] for name in CASES])
        stats = getattr(self, "stats", None)
        if stats is not None:
            # the training rows of the two genome neighbours in the same run
            nb = [_find_rows(self.row_key, self.row_order, of, gene, run,
                             self.ix.n_run)
                  for of in (self.up_gene, self.down_gene)]

        def dev_slice(a, s, e):
            return torch.as_tensor(np.asarray(a[s:e])).long().to(dev)

        out = np.empty((n, 2))
        for s in range(0, n, EVAL_CHUNK):
            e = s + EVAL_CHUNK
            g = torch.as_tensor(gi[s:e]).long().to(dev)
            r = torch.as_tensor(ri[s:e]).long().to(dev)
            on = self._on(g, r, self.seen_dev)
            known = torch.as_tensor(~bad[s:e]).to(dev)
            on = tuple(o & known for o in on)
            st = (stats.external(g, r, dev_slice(gp, s, e),
                                 dev_slice(gc, s, e),
                                 [dev_slice(a, s, e) for a in nb], on)
                  if stats is not None else None)
            outs = []
            for net in nets:
                with self._amp():
                    eta, log_r, _ = net(g, r, *on, stats=st)
                outs.append((eta.float(), log_r.float()))
            eta, log_r = _mix(*zip(*outs, strict=True))
            eta = eta.float().cpu().numpy() + off[s:e]
            log_r = log_r.float().cpu().numpy() + shifts[case[s:e]]
            out[s:e, 0] = np.exp(np.clip(eta, -12, 18))
            out[s:e, 1] = np.exp(np.clip(log_r, -7, 12))
        return out


class Solver(GeneCompetSolver):

    name = "transformer"

    # one benchopt parameter per field of TConfig, so the local harness and
    # the ablations can set any of them (--param norm=rmsnorm). The default
    # values are the submitted configuration, PRESET on the 2017 defaults.
    parameters = {f.name: [PRESET.get(f.name, f.default)]
                  for f in fields(TConfig)}

    def get_model(self, meta):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = TConfig(**{f.name: getattr(self, f.name)
                         for f in fields(TConfig)})
        return TransformerModel(meta, cfg, device=device)

    def fit(self, model, train_df, samples_df, genomes_dir):
        model.fit(train_df, samples_df, genomes_dir)
