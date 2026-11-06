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

    def tokens(self, g, r, gene_on, run_on, prj_on, cond_on, glm=None):
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
        x = torch.cat([part.to(nums.dtype) for part in parts], dim=1)
        return self.drop(self.slots(x))

    def biases(self, g, r, gene_on, run_on, prj_on):
        return self.bias(g, r, self.genome_of_gene[g], self.prj_of_run[r],
                         gene_on, run_on, prj_on)

    def forward(self, g, r, gene_on, run_on, prj_on, cond_on):
        """log mu without the offset, log sigma, auxiliary loss."""
        glm = (self.glm_terms(g, r, gene_on, run_on, prj_on, cond_on)
               if self.glm_mode != "off" else None)
        x = self.tokens(g, r, gene_on, run_on, prj_on, cond_on, glm)
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

    # --- data ---------------------------------------------------------

    def prepare(self, train_df, samples_df):
        """Index, pairs, per gene and per run tables, before any training.

        Returns the gene id, run id, count and offset of each training row.
        Also used by visualisation/build_arch.py to build the real model.
        With ``BACTERIA_KEEP_PREPARED=1`` (tools/ablate.py) the tables are
        kept for the next fit on the same rows. The platform fits once.
        """
        key = (id(train_df), len(train_df), id(samples_df))
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
                 "field_emb", "slots.", "numbers.", "bias."))
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
            with self._amp():
                eta, log_r, _ = net(g, r, *self._on(g, r, seen_dev))
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
            with self._amp():
                eta, log_r, aux = net(g, r, *on)
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

    def fit(self, train_df, samples_df, genomes_dir):
        t0 = time.time()
        cfg, dev = self.cfg, self.device
        budget = cfg.max_minutes * 60
        torch.manual_seed(cfg.seed)
        gene, run, y, off = self.prepare(train_df, samples_df)
        ix = self.ix

        # 1. Internal holdouts built like the regions C, B and A. B also
        #    holds whole projects out: 60 % of the region B rows have a new
        #    project and a new condition, the case the GLM calibrated worst.
        hold_gene = _hit(self.family, 20, seed=0)[gene]
        cond = ix.cond_of_run
        run_out = ix.has_sheet & (_hit(np.maximum(cond, 0), 25, seed=1)
                                  | _hit(ix.exp_of_run, 50, seed=2)
                                  | _hit(ix.prj_of_run, 40, seed=3))
        hold_run = run_out[run]
        free = ~hold_gene & ~hold_run
        block_of_run = np.where(cond >= 0, cond,
                                ix.n_cond + ix.prj_of_run).astype(np.int32)
        hold_block = _hash_hit(gene, block_of_run[run], 50) & free
        hold_pair = _hash_hit(gene, run, 50) & free & ~hold_block
        fit_rows = ~(hold_gene | hold_run | hold_block | hold_pair)
        groups = {"A": hold_block | hold_pair, "B": hold_run & ~hold_gene,
                  "C": hold_gene & ~hold_run, "D": hold_gene & hold_run}
        self.log("holdout rows: " + ", ".join(
            f"{k} {int(v.sum()):,}" for k, v in groups.items())
            + f", fit {int(fit_rows.sum()):,}, {_ram()}")
        hold_gene = hold_run = hold_block = hold_pair = free = None

        # 2. Device tables and the network
        self.t = {"gene": torch.as_tensor(gene).to(dev),
                  "run": torch.as_tensor(run).to(dev),
                  "y": torch.as_tensor(y).to(dev),
                  "off": torch.as_tensor(off).to(dev)}
        self.family_dev = torch.as_tensor(self.family).long().to(dev)
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
        out = np.empty((n, 2))
        for s in range(0, n, EVAL_CHUNK):
            e = s + EVAL_CHUNK
            g = torch.as_tensor(gi[s:e]).long().to(dev)
            r = torch.as_tensor(ri[s:e]).long().to(dev)
            on = self._on(g, r, self.seen_dev)
            known = torch.as_tensor(~bad[s:e]).to(dev)
            on = tuple(o & known for o in on)
            outs = []
            for net in nets:
                with self._amp():
                    eta, log_r, _ = net(g, r, *on)
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
