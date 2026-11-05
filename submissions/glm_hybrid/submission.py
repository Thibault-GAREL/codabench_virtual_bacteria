"""GLM for the genes seen in training, hybrid ESM-2 for the new ones.

Platform scores of the two models alone (public test, 2026-11-05):

    model     A      B      C      challenge
    GLM       7.65   7.71   8.32   7.89
    hybrid    7.80   7.98   7.77   7.85
    this      7.65   7.71   7.77   7.71 expected

The towers of the hybrid stand in for a gene the GLM has never seen (regions C
and D) and add noise to the others, whose own GLM effects already say more
than the protein. The local harness shows the same split (hybrid minus GLM:
A +0.08, B +0.24, C -0.53 locally, A +0.15, B +0.27, C -0.55 on the
platform).

Each row goes to one model by a fixed rule, the gene seen or not in training,
never by the internal holdouts: they rank the hybrid first on A and B, which
the test contradicts.

Both models are copied unchanged from ``submissions/hybrid_esm`` and
``submissions/glm_nb`` (their docstrings describe them), and trained one
after the other on every row: the hybrid first, so that its towers start from
the same random draw as in its own submission, then the GLM. About 1,050 s on
the T4 of the platform (805 s + 245 s).
"""

import gc as _gc
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import gammaln

from benchmark_utils.base_solver import GeneCompetSolver
from benchmark_utils.nb_loss import nb_nll

CHUNK = 8_000_000
# Rows per forward / backward pass on the GPU. 2 M keeps a 6 GB card (and the
# 16 GB T4 of the platform) far from spilling into shared memory.
GPU_CHUNK = 2_000_000
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
# Prior standard deviation of each effect, on the log scale. It sets the
# strength of its L2 penalty.
PRIOR_SD = {"a_genome": 3.0, "a_gene": 3.0, "b_project": 1.0,
            "b_condition": 1.0, "b_run": 1.0, "c_gp": 1.0, "c_gc": 1.0,
            "rho_genome": 1.0, "rho_gene": 1.0}
# Effects with a last "dummy" slot pinned at 0, used by the rows that have no
# such level (no condition, or an interaction seen in fewer than 2 rows)
DUMMY = ("b_condition", "c_gp", "c_gc")
QC_MIN_CORR = 0.5
QC_MAX_ZERO = 0.5
FEATURE_FILE = "gene_features.npz"
# Environment variable that points at the feature file in local runs
FEATURE_ENV = "BACTERIA_GENE_FEATURES"
GENOME_DIM = 8
RUN_CATEGORIES = ["medium_category", "oxygen_condition", "growth_mode",
                  "growth_phase", "perturbation_category", "is_wild_type",
                  "has_perturbation", "condition_confidence"]
RUN_NAMES = ["medium_name", "perturbation_compound"]
RUN_NUMBERS = ["temperature_celsius", "ph", "agitation_rpm", "od600"]
TOP_NAMES = 40


def _clean_name(value):
    """Lower case, no parenthesis, single spaces."""
    if not isinstance(value, str):
        return "unknown"
    value = re.sub(r"\([^)]*\)", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip() or "unknown"


def _run_features(runs):
    """Condition features of each run: one-hots, numbers, missing flags."""
    blocks = []
    for col in RUN_CATEGORIES:
        if col in runs:
            values = runs[col].astype(str).str.lower().where(
                runs[col].notna(), "unknown")
            blocks.append(pd.get_dummies(values, prefix=col, dtype=float))
    for col in RUN_NAMES:
        if col in runs:
            values = runs[col].map(_clean_name)
            top = values[values != "unknown"].value_counts().index[:TOP_NAMES]
            values = values.where(values.isin(top), "other")
            blocks.append(pd.get_dummies(values, prefix=col, dtype=float))
    for col in RUN_NUMBERS:
        if col in runs:
            v = pd.to_numeric(runs[col], errors="coerce")
            z = (v - v.mean()) / (v.std() if v.std() > 0 else 1.0)
            blocks.append(pd.DataFrame({f"{col}_z": z.fillna(0.0),
                                        f"{col}_known": v.notna().astype(
                                            float)}))
    if not blocks:
        return np.zeros((len(runs), 1), dtype=np.float32)
    return pd.concat(blocks, axis=1).to_numpy(dtype=np.float32)


def _gene_features(genes, submission_dir, log):
    """Gene tower inputs aligned on the gene table, None without the file."""
    paths = [Path(submission_dir) / FEATURE_FILE] if submission_dir else []
    if os.environ.get(FEATURE_ENV):
        paths.append(Path(os.environ[FEATURE_ENV]))
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        log(f"no {FEATURE_FILE}, the gene tower sees the genome only")
        return None
    z = np.load(path)
    keys = pd.Index(np.char.add(np.char.add(z["genome_id"], b"|"),
                                z["feature_id"]).astype(str))
    own = (genes["genome_id"].astype(str) + "|"
           + genes["feature_id"].astype(str))
    row = keys.get_indexer(own)
    found = row >= 0
    numeric = np.zeros((len(genes), z["numeric"].shape[1]), dtype=np.float32)
    numeric[found] = z["numeric"][row[found]]
    prot = np.full(len(genes), -1, dtype=np.int64)
    prot[found] = z["protein_idx"][row[found]]
    esm = z["esm"].astype(np.float32)
    emb = np.zeros((len(genes), esm.shape[1]), dtype=np.float32)
    has = prot >= 0
    emb[has] = esm[prot[has]]
    log(f"gene features {path.name}: {int(found.sum()):,} of {len(genes):,} "
        f"genes, {numeric.shape[1]} signals, ESM {esm.shape[1]} dimensions, "
        f"{int(has.sum()):,} with a protein")
    return np.concatenate([numeric, emb, has[:, None].astype(np.float32),
                           found[:, None].astype(np.float32)], axis=1)


class _Towers(nn.Module):
    """Gene tower and run tower, each from its features and the genome."""

    def __init__(self, n_genome, gene_dim, run_dim, dim, hidden):
        super().__init__()
        self.dim = dim
        self.genome = nn.Embedding(n_genome, GENOME_DIM)
        self.gene = nn.Sequential(
            nn.Linear(gene_dim + GENOME_DIM, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 2 + dim))
        self.run = nn.Sequential(
            nn.Linear(run_dim + GENOME_DIM, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1 + dim))
        # start from the GLM: every tower output at 0
        for head in (self.gene[-1], self.run[-1]):
            nn.init.zeros_(head.bias)
            nn.init.normal_(head.weight, std=1e-3)

    def forward(self, x_gene, genome_of_gene, x_run, genome_of_run):
        g = self.gene(torch.cat([x_gene, self.genome(genome_of_gene)], 1))
        r = self.run(torch.cat([x_run, self.genome(genome_of_run)], 1))
        return g, r


class _Table(nn.Module):
    """One GLM effect: a free value per level, read with index_select.

    index_select rather than weight[idx]: on rows sorted by genome and run,
    the backward of weight[idx] is 26 times slower.
    """

    def __init__(self, values):
        super().__init__()
        self.weight = nn.Parameter(torch.as_tensor(np.asarray(values),
                                                   dtype=torch.float32))

    def forward(self, idx=None):
        if idx is None:
            return self.weight
        return torch.index_select(self.weight, 0, idx)


class _Net(nn.Module):
    """The whole model as modules: the GLM effect tables and the two towers.

    Training computes the towers once per step for every gene and run, then
    calls ``rows`` chunk by chunk. ``forward`` chains both for a few rows with
    every level kept, as at prediction time (the visualiser measures it).
    """

    def __init__(self, init, towers):
        super().__init__()
        self.glm = nn.ModuleDict({k: _Table(v) for k, v in init.items()
                                  if not k.startswith("tower")})
        self.towers = towers

    def rows(self, gt, rt, g, r, G, p, c, gp, gc, off, mg, mc, mr, scale):
        """log mu and log sigma of a chunk of rows, tower outputs gathered."""
        t = self.glm
        eta = (off + t["b0"]() + t["a_genome"](G)
               + t["a_gene"](g) * mg
               + t["b_project"](p)
               + t["b_condition"](c) * mc
               + t["b_run"](r) * mr
               + t["c_gp"](gp) * mg
               + t["c_gc"](gc) * mg * mc
               + gt[:, 0] + rt[:, 0]
               + (gt[:, 2:] * rt[:, 1:]).sum(1) * scale)
        log_r = (t["rho0"]() + t["rho_genome"](G)
                 + t["rho_gene"](g) * mg + gt[:, 1])
        return eta, log_r

    def forward(self, x_gene, genome_of_gene, x_run, genome_of_run,
                g, r, G, p, c, gp, gc, off):
        tg, tr = self.towers(x_gene, genome_of_gene, x_run, genome_of_run)
        keep = torch.ones_like(off)
        return self.rows(_take(tg, g), _take(tr, r), g, r, G, p, c, gp, gc,
                         off, keep, keep, keep,
                         1.0 / np.sqrt(self.towers.dim))


def _print_log(message):
    print(f"{time.strftime('%H:%M:%S')} [glm-hybrid] {message}", flush=True)


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


def _take(table, idx):
    """``table[idx]`` through index_select, whose backward is an index_add."""
    return torch.index_select(table, 0, idx)


def _nb_nll_torch(y, log_mu, log_r):
    """NB negative log-likelihood without the constant lgamma(y + 1)."""
    r = torch.exp(log_r)
    log_rm = torch.logaddexp(log_r, log_mu)
    ll = (torch.lgamma(y + r) - torch.lgamma(r)
          + r * (log_r - log_rm) + y * (log_mu - log_rm))
    return -ll


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

        fields = [c for c in CONDITION_FIELDS if c in runs]
        if "condition_confidence" in runs:
            sheet = runs["condition_confidence"].notna().to_numpy()
        else:
            sheet = runs[fields].notna().any(axis=1).to_numpy()
        key = runs["genome_id"].astype(str)
        for column in fields:
            # not astype(str): pandas 3 keeps a missing value missing, and a
            # single empty field would then blank the whole key (26 conditions
            # left instead of 3,664)
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
            code = None
            self.idx = idx.astype(np.int32)
        else:
            self.idx = np.full(len(gene), -1, dtype=np.int32)
            idx, uniques = pd.factorize(code[valid])
            code = None
            self.idx[valid] = idx
        idx = None
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


class HybridModel:
    """Fit the NB GLM with its two towers and predict ``(mu, sigma)``."""

    def __init__(self, meta, steps=200, refit_steps=80, lr=0.05,
                 gene_project=True, gene_condition=True, outlier_weight=1.0,
                 dim=32, hidden=256, lr_tower=3e-3, drop_gene=0.1,
                 drop_cond=0.1, drop_run=0.05, device="cpu"):
        self.genes = meta["feature_annotations"]
        self.submission_dir = meta.get("submission_dir")
        self.steps, self.refit_steps, self.lr = steps, refit_steps, lr
        self.gene_project, self.gene_condition = gene_project, gene_condition
        self.outlier_weight = outlier_weight
        self.dim, self.hidden, self.lr_tower = dim, hidden, lr_tower
        self.drop_gene, self.drop_cond = drop_gene, drop_cond
        self.drop_run = drop_run
        self.device = device
        # Hooks the local harness replaces with RunLogger methods
        self.log = _print_log
        self.metric = _no_metric
        self.track = lambda iterable, desc="train", total=None: iterable
        # factory(module) -> LayerStats, set by the harness to log layer/...
        self.layer_stats = None
        self.params = None

    # --- data ---------------------------------------------------------

    def _run_weights(self, gene, run, y, off, rows):
        """Weight of each row, ``outlier_weight`` for outlier runs."""
        if self.outlier_weight == 1.0:
            return None
        ix = self.ix
        z = np.log(y[rows] + 0.5) - off[rows]
        g, r = gene[rows], run[rows]
        n = np.bincount(r, minlength=ix.n_run).astype(np.float64)
        zeros = np.bincount(r, weights=y[rows] == 0, minlength=ix.n_run)
        gm = (np.bincount(g, weights=z, minlength=ix.n_gene)
              / np.maximum(np.bincount(g, minlength=ix.n_gene), 1))[g]
        with np.errstate(invalid="ignore", divide="ignore"):
            sx = np.bincount(r, z, ix.n_run) / n
            sm = np.bincount(r, gm, ix.n_run) / n
            cov = np.bincount(r, z * gm, ix.n_run) / n - sx * sm
            vx = np.bincount(r, z * z, ix.n_run) / n - sx ** 2
            vm = np.bincount(r, gm * gm, ix.n_run) / n - sm ** 2
            corr = cov / np.sqrt(vx * vm)
            bad = (n > 0) & ((corr < QC_MIN_CORR) | (zeros / n > QC_MAX_ZERO))
        self.log(f"outlier runs: {int(bad.sum())}, weight "
                 f"{self.outlier_weight}")
        return np.where(bad[run], self.outlier_weight, 1.0).astype(np.float32)

    def _support(self, gene, run, rows):
        """Levels seen in ``rows``: genes, runs, projects, conditions, pairs."""
        ix = self.ix
        g, r = gene[rows], run[rows]
        cond = ix.cond_of_run[r]
        gp = self.gp.idx[rows]
        gc = self.gc.idx[rows]
        return {
            "gene": np.bincount(g, minlength=ix.n_gene) > 0,
            "run": np.bincount(r, minlength=ix.n_run) > 0,
            "prj": np.bincount(ix.prj_of_run[r], minlength=ix.n_prj) > 0,
            "cond": np.bincount(cond[cond >= 0], minlength=ix.n_cond) > 0,
            "gp": (np.bincount(gp, minlength=max(self.gp.n, 1)) >= 2)
            & self.gene_project,
            "gc": (np.bincount(gc[gc >= 0], minlength=max(self.gc.n, 1)) >= 2)
            & self.gene_condition,
        }

    # --- training -----------------------------------------------------

    def _init_params(self, gene, run, y, off, rows):
        """Moment estimates on the log scale, a warm start for Adam."""
        ix = self.ix
        z = np.log(y[rows] + 0.5) - off[rows]
        g, r = gene[rows], run[rows]
        b0 = float(z.mean())
        a_gene = (np.bincount(g, z - b0, ix.n_gene)
                  / (np.bincount(g, minlength=ix.n_gene) + 1.0))
        res = z - b0 - a_gene[g]
        b_run = (np.bincount(r, res, ix.n_run)
                 / (np.bincount(r, minlength=ix.n_run) + 1.0))
        zero = np.zeros
        return {"b0": np.array([b0]), "a_genome": zero(ix.n_genome),
                "a_gene": a_gene, "b_project": zero(ix.n_prj),
                "b_condition": zero(ix.n_cond + 1), "b_run": b_run,
                "c_gp": zero(self.gp.n + 1), "c_gc": zero(self.gc.n + 1),
                "rho0": np.array([np.log(5.0)]),
                "rho_genome": zero(ix.n_genome), "rho_gene": zero(ix.n_gene)}

    def _train(self, data, rows, init, steps, stage, offset=0):
        """Adam on the mean NB NLL of ``rows`` plus the L2 penalties.

        ``offset`` numbers the steps after those of an earlier stage, so the
        fit and the refit follow each other on one axis in MLflow.
        """
        dev, ix = self.device, self.ix
        sel = np.flatnonzero(rows)
        n = len(sel)
        support = self._support(data["gene"], data["run"], rows)
        # lgamma(y + 1) is left out of the gradient, added back to the log
        const = sum(float(gammaln(data["y"][sel[s:s + CHUNK]]
                                  .astype(np.float64) + 1).sum())
                    for s in range(0, n, CHUNK)) / n

        def to_dev(name):
            return torch.as_tensor(data[name][sel]).to(dev)

        self.log(f"{stage}: {n:,} rows, {_ram()}")
        t = {name: to_dev(name) for name in ("gene", "run", "y", "off")}
        if data.get("w") is not None:
            t["w"] = to_dev("w")
        # Interaction ids, the dummy slot for the pairs seen in < 2 rows
        for name, pairs in (("gp", self.gp), ("gc", self.gc)):
            idx = torch.as_tensor(pairs.idx[sel]).to(dev).long()
            ok = torch.as_tensor(support[name]).to(dev)
            keep = (idx >= 0) & ok[idx.clamp(min=0)]
            t[name] = torch.where(keep, idx, pairs.n).int()
        genome_of_gene = torch.as_tensor(ix.genome_of_gene).to(dev).long()
        prj_of_run = torch.as_tensor(ix.prj_of_run).to(dev).long()
        cond_of_run = torch.as_tensor(np.where(
            ix.cond_of_run >= 0, ix.cond_of_run, ix.n_cond)).to(dev).long()

        net = _Net(init, self.towers).to(dev)
        P = {k: m.weight for k, m in net.glm.items()}
        towers = net.towers
        # per-layer measures, only when the local harness plugs LayerStats in
        stats = self.layer_stats(net) if self.layer_stats else None
        if stats is not None:
            self.log(stats.describe())
        x_gene = torch.as_tensor(self.x_gene).to(dev)
        x_run = torch.as_tensor(self.x_run).to(dev)
        genome_of_run = torch.as_tensor(np.maximum(ix.genome_of_run, 0)).to(
            dev).long()
        opt = torch.optim.Adam(
            [{"params": list(P.values()), "lr": self.lr, "base": self.lr},
             {"params": towers.parameters(), "lr": self.lr_tower,
              "base": self.lr_tower, "weight_decay": 1e-5}])
        cond_of_cold = np.where(ix.cond_of_run >= 0, ix.cond_of_run, -1)
        if dev != "cpu":
            self.log(f"{stage}: data on {dev}, "
                     f"{torch.cuda.memory_allocated() / 1e9:.2f} GB, {_ram()}")
        scale = 1.0 / np.sqrt(self.dim)
        for step in self.track(range(steps), desc=stage, total=steps):
            decay = 0.1 + 0.45 * (1 + np.cos(np.pi * step / steps))
            for group in opt.param_groups:
                group["lr"] = group["base"] * decay
            opt.zero_grad(set_to_none=True)
            if stats is not None:
                stats.start(step + offset)

            # Level dropout: these genes, conditions and runs lose their own
            # GLM effects for this step, the towers have to stand in
            rng = np.random.default_rng(1000 + step)
            cold_family = rng.random(self.n_family) < self.drop_gene
            cold_gene = cold_family[self.family]
            cold_cond = np.append(rng.random(ix.n_cond) < self.drop_cond,
                                  False)
            cold_run = (rng.random(ix.n_run) < self.drop_run) \
                | cold_cond[cond_of_cold]
            keep_g = torch.as_tensor(~cold_gene).to(dev).float()
            keep_c = torch.as_tensor(~cold_cond).to(dev).float()
            keep_r = torch.as_tensor(~cold_run).to(dev).float()

            # Towers for every gene and run, then gathered row by row
            tg, tr = towers(x_gene, genome_of_gene, x_run, genome_of_run)
            tg_leaf = tg.detach().requires_grad_()
            tr_leaf = tr.detach().requires_grad_()

            total = 0.0
            t_step = time.time()
            for s in range(0, n, GPU_CHUNK):
                e = s + GPU_CHUNK
                g = t["gene"][s:e].long()
                r = t["run"][s:e].long()
                G = genome_of_gene[g]
                c = cond_of_run[r]
                mg, mc, mr = _take(keep_g, g), _take(keep_c, c), \
                    _take(keep_r, r)
                gt, rt = _take(tg_leaf, g), _take(tr_leaf, r)
                eta, log_r = net.rows(
                    gt, rt, g, r, G, prj_of_run[r], c,
                    t["gp"][s:e].long(), t["gc"][s:e].long(),
                    t["off"][s:e], mg, mc, mr, scale)
                nll = _nb_nll_torch(t["y"][s:e], eta, log_r)
                if "w" in t:
                    nll = nll * t["w"][s:e]
                loss = nll.sum() / n
                loss.backward()
                total += float(loss.detach())
            torch.autograd.backward([tg, tr], [tg_leaf.grad, tr_leaf.grad])
            penalty = sum((P[k] ** 2).sum() / (2 * sd ** 2)
                          for k, sd in PRIOR_SD.items()) / n
            penalty.backward()
            if stats is not None:
                stats.before_optimizer()
            opt.step()
            with torch.no_grad():
                for k in DUMMY:
                    P[k][-1] = 0.0
            layer = stats.collect() if stats is not None else {}
            if step == 0:
                gpu = (f", GPU reserved {torch.cuda.memory_reserved() / 1e9:.2f}"
                       " GB" if dev != "cpu" else "")
                self.log(f"{stage}: first step {time.time() - t_step:.1f}s"
                         f"{gpu}, {_ram()}")
            if step % 10 == 0 or step == steps - 1 or layer:
                self.metric(step=step + offset,
                            **{f"{stage}/nll": total + const,
                               f"{stage}/lr": self.lr * decay}, **layer)
        out = {k: v.detach().cpu().numpy().astype(np.float64)
               for k, v in P.items()}
        with torch.no_grad():
            towers.eval()
            tg, tr = towers(x_gene, genome_of_gene, x_run, genome_of_run)
            towers.train()
        out["tower_g"] = tg.cpu().numpy().astype(np.float32)
        out["tower_r"] = tr.cpu().numpy().astype(np.float32)
        if stats is not None:
            stats.remove()
        del t, P, opt, x_gene, x_run, tg, tr, net
        if dev != "cpu":
            torch.cuda.empty_cache()
        return out, support

    # --- prediction core ----------------------------------------------

    def _linear(self, P, seen, gene, genome, run, off, gp=None, gc=None):
        """log mu, base log sigma and case of each row."""
        ix = self.ix
        g_ok = gene >= 0
        gi = np.where(g_ok, gene, 0)
        G = ix.genome_of_gene[gi] if genome is None else np.where(
            g_ok, ix.genome_of_gene[gi], genome)
        G_ok = G >= 0
        Gi = np.where(G_ok, G, 0)
        r_ok = run >= 0
        ri = np.where(r_ok, run, 0)
        p = ix.prj_of_run[ri]
        c = ix.cond_of_run[ri]
        ci = np.where(c >= 0, c, 0)
        sg = g_ok & seen["gene"][gi]
        sr = r_ok & seen["run"][ri]
        sp = r_ok & seen["prj"][p]
        sc = r_ok & (c >= 0) & seen["cond"][ci]
        if gp is None:
            gp = self.gp.lookup(np.where(sg, gi, -1), np.where(r_ok, p, -1))
            gc = self.gc.lookup(np.where(sg, gi, -1), np.where(r_ok, c, -1))
        gpi, gci = np.maximum(gp, 0), np.maximum(gc, 0)
        vgp = (gp >= 0) & seen["gp"][gpi] & sg
        vgc = (gc >= 0) & seen["gc"][gci] & sg
        eta = (off + P["b0"][0] + P["a_genome"][Gi] * G_ok
               + P["a_gene"][gi] * sg + P["b_project"][p] * sp
               + P["b_condition"][ci] * sc + P["b_run"][ri] * sr
               + P["c_gp"][gpi] * vgp + P["c_gc"][gci] * vgc)
        log_r = P["rho0"][0] + P["rho_genome"][Gi] * G_ok \
            + P["rho_gene"][gi] * sg
        # Towers: defined for every gene and run of the annotation tables
        tg, tr = P["tower_g"][gi], P["tower_r"][ri]
        both = (g_ok & r_ok).astype(np.float32)
        eta = eta + (tg[:, 0] * g_ok + tr[:, 0] * r_ok
                     + (tg[:, 2:] * tr[:, 1:]).sum(1) * both
                     / np.sqrt(self.dim))
        log_r = log_r + tg[:, 1] * g_ok
        case = np.select([sg & sr & (vgp | vgc), sg & sr, sg & sc, sg & sp,
                          sg, sr], [0, 1, 2, 3, 4, 5], 6)
        return eta, log_r, case

    # --- fit / predict ------------------------------------------------

    def prepare(self, train_df, samples_df):
        """Index, pairs, tower inputs and towers, before any training.

        Returns the gene id, run id, count and offset of each training row.
        Also used by visualisation/build_arch.py to build the real model.
        """
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
        self.log(f"{len(y):,} rows, {ix.n_gene:,} genes, {ix.n_run:,} runs, "
                 f"{ix.n_prj:,} projects, {ix.n_cond:,} conditions, "
                 f"{self.gp.n:,} gene x project, {self.gc.n:,} gene x "
                 f"condition ({time.time() - t0:.0f}s)")

        # Tower inputs and the towers themselves
        features = _gene_features(self.genes, self.submission_dir, self.log)
        self.x_gene = (features if features is not None
                       else np.zeros((ix.n_gene, 1), dtype=np.float32))
        self.x_run = _run_features(samples_df)
        self.towers = _Towers(ix.n_genome, self.x_gene.shape[1],
                              self.x_run.shape[1], self.dim, self.hidden)
        self.log(f"towers: gene {self.x_gene.shape[1]} inputs, run "
                 f"{self.x_run.shape[1]} inputs, "
                 f"{sum(p.numel() for p in self.towers.parameters()):,} "
                 f"parameters")
        self.family = _gene_families(self.genes)
        self.n_family = int(self.family.max()) + 1
        return gene, run, y, off

    def fit(self, train_df, samples_df, genomes_dir):
        t0 = time.time()
        gene, run, y, off = self.prepare(train_df, samples_df)
        ix = self.ix

        # 1. Internal holdouts built like the regions C, B and A
        hold_gene = _hit(self.family, 20, seed=0)[gene]
        cond = ix.cond_of_run
        run_out = ix.has_sheet & (_hit(np.maximum(cond, 0), 25, seed=1)
                                  | _hit(ix.exp_of_run, 50, seed=2))
        hold_run = run_out[run]
        free = ~hold_gene & ~hold_run
        block_of_run = np.where(cond >= 0, cond,
                                ix.n_cond + ix.prj_of_run).astype(np.int32)
        hold_block = _hash_hit(gene, block_of_run[run], 50) & free
        hold_pair = _hash_hit(gene, run, 50) & free & ~hold_block
        fit_rows = ~(hold_gene | hold_run | hold_block | hold_pair)
        data = {"gene": gene, "run": run, "y": y, "off": off,
                "w": self._run_weights(gene, run, y, off, fit_rows)}
        self.log(f"holdout rows: {int(hold_gene.sum()):,} gene, "
                 f"{int(hold_run.sum()):,} run, "
                 f"{int(hold_block.sum()):,} gene x condition, "
                 f"{int(hold_pair.sum()):,} pair, {_ram()}")
        hold_gene = hold_run = hold_block = hold_pair = free = None
        _gc.collect()

        # 2. Fit without the holdout rows
        init = self._init_params(gene, run, y, off, fit_rows)
        P, seen = self._train(data, fit_rows, init, self.steps, "fit")
        self.log(f"fit done ({time.time() - t0:.0f}s)")

        # 3. One dispersion shift per case, with the challenge metric
        val = np.flatnonzero(~fit_rows)
        parts = [self._linear(P, seen, gene[v], None, run[v], off[v],
                              self.gp.idx[v], self.gc.idx[v])
                 for v in np.array_split(val, max(1, len(val) // 2_000_000))]
        eta, log_r, case = (np.concatenate(x) for x in zip(*parts))
        parts = None
        self.delta = self._calibrate(y[val], eta, log_r, case)

        # 4. Refit on every row, warm started
        all_rows = np.ones(len(y), dtype=bool)
        data["w"] = self._run_weights(gene, run, y, off, all_rows)
        self.params, self.seen = self._train(data, all_rows, P,
                                             self.refit_steps, "refit",
                                             offset=self.steps)
        self.metric(**self.holdout)
        self.log(f"fit total {time.time() - t0:.0f}s")
        return self

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
        # sent to the tracker once the fit is over, to keep the progress
        # lines short
        self.holdout = {**{f"holdout/nb_nll_{k}": v
                           for k, v in scores.items()},
                        **{f"holdout/delta_{k}": v for k, v in delta.items()}}
        return delta

    def predict(self, test_df, samples_df, genomes_dir):
        n = len(test_df)
        if self.params is None:
            return np.column_stack([np.ones(n), np.ones(n)])
        gene, genome, run = self.ix.rows(test_df)
        off = (test_df["log_length_ratio"].to_numpy(dtype=np.float64)
               + test_df["log_size_factor"].to_numpy(dtype=np.float64))
        off = np.nan_to_num(off)
        shifts = np.array([self.delta[name] for name in CASES])
        out = np.empty((n, 2))
        for s in range(0, n, 1_000_000):
            e = s + 1_000_000
            eta, log_r, case = self._linear(self.params, self.seen,
                                            gene[s:e], genome[s:e],
                                            run[s:e], off[s:e])
            out[s:e, 0] = np.exp(np.clip(eta, -12, 18))
            out[s:e, 1] = np.exp(np.clip(log_r + shifts[case], -7, 12))
        return out


class GLMModel:
    """Fit the NB GLM and predict ``(mu, sigma)``."""

    def __init__(self, meta, steps=150, refit_steps=60, lr=0.05,
                 gene_project=True, gene_condition=True, outlier_weight=1.0,
                 device="cpu"):
        self.genes = meta["feature_annotations"]
        self.steps, self.refit_steps, self.lr = steps, refit_steps, lr
        self.gene_project, self.gene_condition = gene_project, gene_condition
        self.outlier_weight = outlier_weight
        self.device = device
        # Hooks the local harness replaces with RunLogger methods
        self.log = _print_log
        self.metric = _no_metric
        self.track = lambda iterable, desc="train", total=None: iterable
        self.params = None

    # --- data ---------------------------------------------------------

    def _run_weights(self, gene, run, y, off, rows):
        """Weight of each row, ``outlier_weight`` for outlier runs."""
        if self.outlier_weight == 1.0:
            return None
        ix = self.ix
        z = np.log(y[rows] + 0.5) - off[rows]
        g, r = gene[rows], run[rows]
        n = np.bincount(r, minlength=ix.n_run).astype(np.float64)
        zeros = np.bincount(r, weights=y[rows] == 0, minlength=ix.n_run)
        gm = (np.bincount(g, weights=z, minlength=ix.n_gene)
              / np.maximum(np.bincount(g, minlength=ix.n_gene), 1))[g]
        with np.errstate(invalid="ignore", divide="ignore"):
            sx = np.bincount(r, z, ix.n_run) / n
            sm = np.bincount(r, gm, ix.n_run) / n
            cov = np.bincount(r, z * gm, ix.n_run) / n - sx * sm
            vx = np.bincount(r, z * z, ix.n_run) / n - sx ** 2
            vm = np.bincount(r, gm * gm, ix.n_run) / n - sm ** 2
            corr = cov / np.sqrt(vx * vm)
            bad = (n > 0) & ((corr < QC_MIN_CORR) | (zeros / n > QC_MAX_ZERO))
        self.log(f"outlier runs: {int(bad.sum())}, weight "
                 f"{self.outlier_weight}")
        return np.where(bad[run], self.outlier_weight, 1.0).astype(np.float32)

    def _support(self, gene, run, rows):
        """Levels seen in ``rows``: genes, runs, projects, conditions, pairs."""
        ix = self.ix
        g, r = gene[rows], run[rows]
        cond = ix.cond_of_run[r]
        gp = self.gp.idx[rows]
        gc = self.gc.idx[rows]
        return {
            "gene": np.bincount(g, minlength=ix.n_gene) > 0,
            "run": np.bincount(r, minlength=ix.n_run) > 0,
            "prj": np.bincount(ix.prj_of_run[r], minlength=ix.n_prj) > 0,
            "cond": np.bincount(cond[cond >= 0], minlength=ix.n_cond) > 0,
            "gp": (np.bincount(gp, minlength=max(self.gp.n, 1)) >= 2)
            & self.gene_project,
            "gc": (np.bincount(gc[gc >= 0], minlength=max(self.gc.n, 1)) >= 2)
            & self.gene_condition,
        }

    # --- training -----------------------------------------------------

    def _init_params(self, gene, run, y, off, rows):
        """Moment estimates on the log scale, a warm start for Adam."""
        ix = self.ix
        z = np.log(y[rows] + 0.5) - off[rows]
        g, r = gene[rows], run[rows]
        b0 = float(z.mean())
        a_gene = (np.bincount(g, z - b0, ix.n_gene)
                  / (np.bincount(g, minlength=ix.n_gene) + 1.0))
        res = z - b0 - a_gene[g]
        b_run = (np.bincount(r, res, ix.n_run)
                 / (np.bincount(r, minlength=ix.n_run) + 1.0))
        zero = np.zeros
        return {"b0": np.array([b0]), "a_genome": zero(ix.n_genome),
                "a_gene": a_gene, "b_project": zero(ix.n_prj),
                "b_condition": zero(ix.n_cond + 1), "b_run": b_run,
                "c_gp": zero(self.gp.n + 1), "c_gc": zero(self.gc.n + 1),
                "rho0": np.array([np.log(5.0)]),
                "rho_genome": zero(ix.n_genome), "rho_gene": zero(ix.n_gene)}

    def _train(self, data, rows, init, steps, stage):
        """Adam on the mean NB NLL of ``rows`` plus the L2 penalties."""
        dev, ix = self.device, self.ix
        sel = np.flatnonzero(rows)
        n = len(sel)
        support = self._support(data["gene"], data["run"], rows)
        # lgamma(y + 1) is left out of the gradient, added back to the log
        const = sum(float(gammaln(data["y"][sel[s:s + CHUNK]]
                                  .astype(np.float64) + 1).sum())
                    for s in range(0, n, CHUNK)) / n

        def to_dev(name):
            return torch.as_tensor(data[name][sel]).to(dev)

        self.log(f"{stage}: {n:,} rows, {_ram()}")
        t = {name: to_dev(name) for name in ("gene", "run", "y", "off")}
        if data.get("w") is not None:
            t["w"] = to_dev("w")
        # Interaction ids, the dummy slot for the pairs seen in < 2 rows
        for name, pairs in (("gp", self.gp), ("gc", self.gc)):
            idx = torch.as_tensor(pairs.idx[sel]).to(dev).long()
            ok = torch.as_tensor(support[name]).to(dev)
            keep = (idx >= 0) & ok[idx.clamp(min=0)]
            t[name] = torch.where(keep, idx, pairs.n).int()
        genome_of_gene = torch.as_tensor(ix.genome_of_gene).to(dev).long()
        prj_of_run = torch.as_tensor(ix.prj_of_run).to(dev).long()
        cond_of_run = torch.as_tensor(np.where(
            ix.cond_of_run >= 0, ix.cond_of_run, ix.n_cond)).to(dev).long()

        P = {k: torch.tensor(v, dtype=torch.float32, device=dev,
                             requires_grad=True) for k, v in init.items()}
        opt = torch.optim.Adam(list(P.values()), lr=self.lr)
        if dev != "cpu":
            self.log(f"{stage}: data on {dev}, "
                     f"{torch.cuda.memory_allocated() / 1e9:.2f} GB, {_ram()}")
        for step in self.track(range(steps), desc=stage, total=steps):
            lr = self.lr * (0.1 + 0.45 * (1 + np.cos(np.pi * step / steps)))
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            total = 0.0
            t_step = time.time()
            for s in range(0, n, GPU_CHUNK):
                e = s + GPU_CHUNK
                # index_select, not P[k][idx]: on rows sorted by genome and
                # run, the backward of P[k][idx] is 26 times slower
                g = t["gene"][s:e].long()
                r = t["run"][s:e].long()
                G = genome_of_gene[g]
                eta = (t["off"][s:e] + P["b0"] + _take(P["a_genome"], G)
                       + _take(P["a_gene"], g)
                       + _take(P["b_project"], prj_of_run[r])
                       + _take(P["b_condition"], cond_of_run[r])
                       + _take(P["b_run"], r)
                       + _take(P["c_gp"], t["gp"][s:e].long())
                       + _take(P["c_gc"], t["gc"][s:e].long()))
                log_r = (P["rho0"] + _take(P["rho_genome"], G)
                         + _take(P["rho_gene"], g))
                nll = _nb_nll_torch(t["y"][s:e], eta, log_r)
                if "w" in t:
                    nll = nll * t["w"][s:e]
                loss = nll.sum() / n
                loss.backward()
                total += float(loss.detach())
            penalty = sum((P[k] ** 2).sum() / (2 * sd ** 2)
                          for k, sd in PRIOR_SD.items()) / n
            penalty.backward()
            opt.step()
            with torch.no_grad():
                for k in DUMMY:
                    P[k][-1] = 0.0
            if step == 0:
                gpu = (f", GPU reserved {torch.cuda.memory_reserved() / 1e9:.2f}"
                       " GB" if dev != "cpu" else "")
                self.log(f"{stage}: first step {time.time() - t_step:.1f}s"
                         f"{gpu}, {_ram()}")
            if step % 10 == 0 or step == steps - 1:
                self.metric(step=step, **{f"{stage}/nll": total + const,
                                          f"{stage}/lr": lr})
        out = {k: v.detach().cpu().numpy().astype(np.float64)
               for k, v in P.items()}
        del t, P, opt
        if dev != "cpu":
            torch.cuda.empty_cache()
        return out, support

    # --- prediction core ----------------------------------------------

    def _linear(self, P, seen, gene, genome, run, off, gp=None, gc=None):
        """log mu, base log sigma and case of each row."""
        ix = self.ix
        g_ok = gene >= 0
        gi = np.where(g_ok, gene, 0)
        G = ix.genome_of_gene[gi] if genome is None else np.where(
            g_ok, ix.genome_of_gene[gi], genome)
        G_ok = G >= 0
        Gi = np.where(G_ok, G, 0)
        r_ok = run >= 0
        ri = np.where(r_ok, run, 0)
        p = ix.prj_of_run[ri]
        c = ix.cond_of_run[ri]
        ci = np.where(c >= 0, c, 0)
        sg = g_ok & seen["gene"][gi]
        sr = r_ok & seen["run"][ri]
        sp = r_ok & seen["prj"][p]
        sc = r_ok & (c >= 0) & seen["cond"][ci]
        if gp is None:
            gp = self.gp.lookup(np.where(sg, gi, -1), np.where(r_ok, p, -1))
            gc = self.gc.lookup(np.where(sg, gi, -1), np.where(r_ok, c, -1))
        gpi, gci = np.maximum(gp, 0), np.maximum(gc, 0)
        vgp = (gp >= 0) & seen["gp"][gpi] & sg
        vgc = (gc >= 0) & seen["gc"][gci] & sg
        eta = (off + P["b0"][0] + P["a_genome"][Gi] * G_ok
               + P["a_gene"][gi] * sg + P["b_project"][p] * sp
               + P["b_condition"][ci] * sc + P["b_run"][ri] * sr
               + P["c_gp"][gpi] * vgp + P["c_gc"][gci] * vgc)
        log_r = P["rho0"][0] + P["rho_genome"][Gi] * G_ok \
            + P["rho_gene"][gi] * sg
        case = np.select([sg & sr & (vgp | vgc), sg & sr, sg & sc, sg & sp,
                          sg, sr], [0, 1, 2, 3, 4, 5], 6)
        return eta, log_r, case

    # --- fit / predict ------------------------------------------------

    def fit(self, train_df, samples_df, genomes_dir):
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
        self.log(f"{len(y):,} rows, {ix.n_gene:,} genes, {ix.n_run:,} runs, "
                 f"{ix.n_prj:,} projects, {ix.n_cond:,} conditions, "
                 f"{self.gp.n:,} gene x project, {self.gc.n:,} gene x "
                 f"condition ({time.time() - t0:.0f}s)")

        # 1. Internal holdouts built like the regions C, B and A
        hold_gene = _hit(_gene_families(self.genes), 20, seed=0)[gene]
        cond = ix.cond_of_run
        run_out = ix.has_sheet & (_hit(np.maximum(cond, 0), 25, seed=1)
                                  | _hit(ix.exp_of_run, 50, seed=2))
        hold_run = run_out[run]
        free = ~hold_gene & ~hold_run
        block_of_run = np.where(cond >= 0, cond,
                                ix.n_cond + ix.prj_of_run).astype(np.int32)
        hold_block = _hash_hit(gene, block_of_run[run], 50) & free
        hold_pair = _hash_hit(gene, run, 50) & free & ~hold_block
        fit_rows = ~(hold_gene | hold_run | hold_block | hold_pair)
        data = {"gene": gene, "run": run, "y": y, "off": off,
                "w": self._run_weights(gene, run, y, off, fit_rows)}
        self.log(f"holdout rows: {int(hold_gene.sum()):,} gene, "
                 f"{int(hold_run.sum()):,} run, "
                 f"{int(hold_block.sum()):,} gene x condition, "
                 f"{int(hold_pair.sum()):,} pair, {_ram()}")
        hold_gene = hold_run = hold_block = hold_pair = free = None
        _gc.collect()

        # 2. Fit without the holdout rows
        init = self._init_params(gene, run, y, off, fit_rows)
        P, seen = self._train(data, fit_rows, init, self.steps, "fit")
        self.log(f"fit done ({time.time() - t0:.0f}s)")

        # 3. One dispersion shift per case, with the challenge metric
        val = np.flatnonzero(~fit_rows)
        parts = [self._linear(P, seen, gene[v], None, run[v], off[v],
                              self.gp.idx[v], self.gc.idx[v])
                 for v in np.array_split(val, max(1, len(val) // 2_000_000))]
        eta, log_r, case = (np.concatenate(x) for x in zip(*parts))
        parts = None
        self.delta = self._calibrate(y[val], eta, log_r, case)

        # 4. Refit on every row, warm started
        all_rows = np.ones(len(y), dtype=bool)
        data["w"] = self._run_weights(gene, run, y, off, all_rows)
        self.params, self.seen = self._train(data, all_rows, P,
                                             self.refit_steps, "refit")
        self.metric(**self.holdout)
        self.log(f"fit total {time.time() - t0:.0f}s")
        return self

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
        # sent to the tracker once the fit is over, to keep the progress
        # lines short
        self.holdout = {**{f"holdout/nb_nll_{k}": v
                           for k, v in scores.items()},
                        **{f"holdout/delta_{k}": v for k, v in delta.items()}}
        return delta

    def predict(self, test_df, samples_df, genomes_dir):
        n = len(test_df)
        if self.params is None:
            return np.column_stack([np.ones(n), np.ones(n)])
        gene, genome, run = self.ix.rows(test_df)
        off = (test_df["log_length_ratio"].to_numpy(dtype=np.float64)
               + test_df["log_size_factor"].to_numpy(dtype=np.float64))
        off = np.nan_to_num(off)
        eta, log_r, case = self._linear(self.params, self.seen, gene, genome,
                                        run, off)
        shift = np.array([self.delta[name] for name in CASES])[case]
        mu = np.exp(np.clip(eta, -12, 18))
        sigma = np.exp(np.clip(log_r + shift, -7, 12))
        return np.column_stack([mu, sigma])


def _release(model):
    """Drops what only the training needed, before the next model trains."""
    for pairs in (model.gp, model.gc):
        pairs.idx = None
    for name in ("x_gene", "x_run", "family", "towers"):
        if hasattr(model, name):
            setattr(model, name, None)
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class CombinedModel:
    """The hybrid for the genes unseen in training, the GLM for the others."""

    def __init__(self, hybrid, glm):
        self.hybrid, self.glm = hybrid, glm
        # Hooks the local harness replaces with RunLogger methods, handed down
        # to both models by fit
        self.log = _print_log
        self.metric = _no_metric
        self.track = lambda iterable, desc="train", total=None: iterable
        self.layer_stats = None

    def _plug(self):
        """The hybrid logs as usual, the GLM under glm/ (same step axis)."""
        hybrid, glm = self.hybrid, self.glm
        log, metric, track = self.log, self.metric, self.track
        hybrid.log = lambda message: log(f"hybrid | {message}")
        hybrid.metric, hybrid.track = metric, track
        hybrid.layer_stats = self.layer_stats
        glm.log = lambda message: log(f"glm    | {message}")
        glm.metric = lambda step=None, show=False, **values: metric(
            step=step, show=show,
            **{f"glm/{k}": v for k, v in values.items()})
        glm.track = lambda iterable, desc="train", total=None: track(
            iterable, desc=f"glm {desc}", total=total)

    def fit(self, train_df, samples_df, genomes_dir):
        t0 = time.time()
        self._plug()
        self.hybrid.fit(train_df, samples_df, genomes_dir)
        _release(self.hybrid)
        self.glm.fit(train_df, samples_df, genomes_dir)
        _release(self.glm)
        self.log(f"both models fitted ({time.time() - t0:.0f}s), {_ram()}")
        return self

    def predict(self, test_df, samples_df, genomes_dir):
        out = self.glm.predict(test_df, samples_df, genomes_dir)
        if self.glm.params is None or self.hybrid.params is None:
            return out
        gene, _, _ = self.glm.ix.rows(test_df)
        new = (gene < 0) | ~self.glm.seen["gene"][np.maximum(gene, 0)]
        self.log(f"predict: {len(new):,} rows, {int(new.sum()):,} of a gene "
                 f"unseen in training (hybrid), the others GLM")
        if new.any():
            out[new] = self.hybrid.predict(test_df.iloc[np.flatnonzero(new)],
                                           samples_df, genomes_dir)
        return out


class Solver(GeneCompetSolver):

    name = "glm-hybrid"

    parameters = {"steps": [200], "refit_steps": [80], "lr": [0.05],
                  "gene_project": [True], "gene_condition": [True],
                  "outlier_weight": [1.0], "dim": [32], "hidden": [256],
                  "lr_tower": [3e-3], "drop_gene": [0.1], "drop_cond": [0.1],
                  "drop_run": [0.05], "glm_steps": [150],
                  "glm_refit_steps": [60]}

    def get_model(self, meta):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)
        hybrid = HybridModel(
            meta, steps=self.steps, refit_steps=self.refit_steps, lr=self.lr,
            gene_project=self.gene_project,
            gene_condition=self.gene_condition,
            outlier_weight=self.outlier_weight, dim=self.dim,
            hidden=self.hidden, lr_tower=self.lr_tower,
            drop_gene=self.drop_gene, drop_cond=self.drop_cond,
            drop_run=self.drop_run, device=device)
        glm = GLMModel(
            meta, steps=self.glm_steps, refit_steps=self.glm_refit_steps,
            lr=self.lr, gene_project=self.gene_project,
            gene_condition=self.gene_condition,
            outlier_weight=self.outlier_weight, device=device)
        return CombinedModel(hybrid, glm)

    def fit(self, model, train_df, samples_df, genomes_dir):
        model.fit(train_df, samples_df, genomes_dir)
