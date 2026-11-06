"""Tests of submissions/transformer/submission.py.

As in Modern_LLM, each fast or modern component is checked against its plain
reference: the fused attention against softmax(QK^T)V, GQA with every KV head
against MHA, RMSNorm and LayerNorm against their formula, a one expert MoE
against its expert, the capped NB loss against the official metric. Then the
whole model fits and predicts on a small synthetic dataset.

    pytest tests -q
"""

import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

from benchmark_utils.nb_loss import nb_nll  # noqa: E402


def _load():
    path = ROOT / "submissions" / "transformer" / "submission.py"
    spec = importlib.util.spec_from_file_location("transformer_sub", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


T = _load()
CUDA = torch.cuda.is_available()


def cfg(**kw):
    return replace(T.TConfig(d_model=32, n_heads=4, n_layers=2), **kw)


# --- loss ----------------------------------------------------------------------

def test_capped_nll_matches_official_metric():
    rng = np.random.default_rng(0)
    n = 5000
    mu = np.exp(rng.uniform(-3, 9, n))
    sigma = np.exp(rng.uniform(-3, 5, n))
    y = rng.negative_binomial(sigma, sigma / (sigma + mu)).astype(np.float64)
    # a few catastrophic rows that hit the cap
    y[:20] = 1e5
    mu[:20] = 1e-3
    ours = T._nb_nll_capped(torch.as_tensor(y), torch.as_tensor(np.log(mu)),
                            torch.as_tensor(np.log(sigma)))
    assert float(ours.max()) == pytest.approx(100.0)
    assert float(ours.mean()) == pytest.approx(nb_nll(y, mu, sigma), rel=1e-9)


# --- attention -----------------------------------------------------------------

@pytest.mark.parametrize("attn", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_sdpa_matches_naive_attention(attn, qk_norm):
    torch.manual_seed(0)
    naive = T._Attention(cfg(attn=attn, qk_norm=qk_norm, sdpa=False)).eval()
    fused = T._Attention(cfg(attn=attn, qk_norm=qk_norm, sdpa=True)).eval()
    fused.load_state_dict(naive.state_dict())
    x = torch.randn(8, 21, 32)
    torch.testing.assert_close(fused(x), naive(x), atol=1e-5, rtol=1e-5)


def test_gqa_with_every_kv_head_is_mha():
    torch.manual_seed(0)
    mha = T._Attention(cfg(attn="mha")).eval()
    gqa = T._Attention(cfg(attn="gqa", n_kv_heads=4)).eval()
    gqa.load_state_dict(mha.state_dict())
    x = torch.randn(4, 21, 32)
    torch.testing.assert_close(gqa(x), mha(x))


def test_kv_heads_shrink_the_projections():
    sizes = {a: sum(p.numel() for p in T._Attention(cfg(attn=a)).parameters())
             for a in ("mha", "gqa", "mqa")}
    assert sizes["mqa"] < sizes["gqa"] < sizes["mha"]


def test_repeat_kv_shares_each_head():
    x = torch.randn(2, 2, 5, 8)
    torch.testing.assert_close(T._repeat_kv(x, 3),
                               x.repeat_interleave(3, dim=1))


def test_softcap_bounds_the_scores():
    torch.manual_seed(0)
    att = T._Attention(cfg(softcap=0.5, sdpa=True)).eval()
    # softcap forces the naive path, even when sdpa is asked for
    assert not att.sdpa
    assert torch.isfinite(att(torch.randn(2, 21, 32) * 100)).all()


# --- norms and feed-forward ----------------------------------------------------

def test_rmsnorm_matches_its_formula():
    norm = T._Norm("rmsnorm", 16)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    x = torch.randn(4, 16)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * norm.weight
    torch.testing.assert_close(norm(x), ref)


def test_layernorm_matches_torch():
    ours, ref = T._Norm("layernorm", 16), torch.nn.LayerNorm(16)
    with torch.no_grad():
        ref.weight.uniform_(0.5, 1.5)
        ref.bias.uniform_(-1, 1)
    ours.load_state_dict(ref.state_dict())
    x = torch.randn(4, 16)
    torch.testing.assert_close(ours(x), ref(x))


def test_dyt_has_no_reduction():
    norm = T._Norm("dyt", 16)
    x = torch.randn(4, 16)
    torch.testing.assert_close(norm(x), torch.tanh(0.5 * x))


def test_gated_ffn_keeps_the_parameter_count():
    def count(kind):
        return sum(p.numel() for p in T._FFN(kind, 64, 4.0).parameters())
    assert abs(count("swiglu") - count("relu")) / count("relu") < 0.02


def test_moe_with_one_expert_is_that_expert():
    torch.manual_seed(0)
    moe = T._MoE(cfg(moe_experts=1, moe_top_k=1, ffn="gelu"))
    x = torch.randn(3, 21, 32)
    torch.testing.assert_close(moe(x), moe.experts[0](x))


def test_moe_aux_loss_is_one_for_a_uniform_router():
    moe = T._MoE(cfg(moe_experts=8, moe_top_k=2))
    with torch.no_grad():
        moe.router.weight.zero_()
    moe(torch.randn(64, 32))
    assert float(moe.aux) == pytest.approx(1.0)


@pytest.mark.parametrize("place", ["post", "pre", "sandwich"])
@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm", "dyt"])
def test_block_placements(place, norm):
    block = T._Block(cfg(norm_place=place, norm=norm, ffn="swiglu"))
    out = block(torch.randn(3, 21, 32))
    assert out.shape == (3, 21, 32) and torch.isfinite(out).all()


# --- schedule ------------------------------------------------------------------

def test_schedules():
    c = T.TConfig(warmup_steps=10)
    noam = [T._lr_factor(c, s, 100) for s in range(100)]
    assert max(noam) == pytest.approx(1.0) and np.argmax(noam) == 9
    cos = replace(c, schedule="cosine")
    assert T._lr_factor(cos, 99, 100) == pytest.approx(0.1, abs=1e-3)
    wsd = replace(c, schedule="wsd")
    assert T._lr_factor(wsd, 50, 100) == 1.0
    assert T._lr_factor(wsd, 99, 100) == pytest.approx(0.1, abs=0.02)


def test_schedule_follows_the_clock_when_time_binds():
    # on time: the schedule reads the real step
    assert T._schedule_step(10, 100, elapsed=5.0, budget=100.0) == 10
    # half the time spent at step 10: the schedule is at step 50
    assert T._schedule_step(10, 100, elapsed=50.0, budget=100.0) == 50
    # never past the last step, even after the deadline
    assert T._schedule_step(10, 100, elapsed=500.0, budget=100.0) == 99
    c = replace(T.TConfig(), schedule="cosine", warmup_steps=5)
    assert T._lr_factor(c, T._schedule_step(10, 100, 100.0, 100.0), 100) \
        == pytest.approx(0.1, abs=1e-3)


def test_solver_defaults_are_the_submitted_configuration():
    from dataclasses import fields
    for f in fields(T.TConfig):
        expected = T.PRESET.get(f.name, f.default)
        assert T.Solver.parameters[f.name] == [expected], f.name
    # the 2017 defaults themselves are untouched, the ablations start there
    assert T.TConfig().norm_place == "post" and T.TConfig().id_bias


# --- the whole network ---------------------------------------------------------

def tiny_tables(n_gene=6, n_run=5):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n_gene, 7)).astype(np.float16)
    x[1] = x[0]                      # genes 0 and 1 look identical
    return {
        "n_gene": n_gene, "n_run": n_run, "n_prj": 2, "n_cond": 3,
        "n_genome": 1, "n_vocab": 9, "x_gene": x,
        "genome_of_gene": np.zeros(n_gene, dtype=np.int64),
        "prj_of_run": np.array([0, 0, 1, 1, 1]),
        "cond_of_run": np.array([0, 1, -1, 2, 2]),
        "run_cat": rng.integers(0, 9, size=(n_run, 3)),
        "run_num": rng.normal(size=(n_run, 4)).astype(np.float32),
        "run_known": (rng.random((n_run, 4)) > 0.3).astype(np.float32),
    }


def tiny_bias(n_gene=6, n_run=5):
    rng = np.random.default_rng(1)
    return {"b0": np.array(1.0), "a_genome": np.zeros(1),
            "a_gene": rng.normal(size=n_gene), "b_run": rng.normal(size=n_run),
            "b_project": np.zeros(2), "rho0": np.array(math.log(5.0)),
            "rho_gene": np.zeros(n_gene)}


def flags(b, value=True):
    return tuple(torch.full((b,), value) for _ in range(4))


def test_head_starts_at_the_id_biases():
    net = T._ExprTransformer(cfg(), tiny_tables(), tiny_bias()).eval()
    g, r = torch.tensor([0, 2, 5]), torch.tensor([1, 3, 4])
    eta, log_r, _ = net(g, r, *flags(3))
    be, br = net.biases(g, r, *[f.float() for f in flags(3)[:3]])
    torch.testing.assert_close(eta, be)
    torch.testing.assert_close(log_r, br)


def test_an_unseen_gene_reads_the_mask_token():
    torch.manual_seed(0)
    net = T._ExprTransformer(cfg(), tiny_tables(), tiny_bias()).eval()
    with torch.no_grad():
        net.head.weight.normal_()     # a head that reads the tokens
    r = torch.tensor([2, 2])
    g = torch.tensor([0, 1])
    hidden, _, _ = net(g, r, torch.tensor([False, False]),
                       *flags(2)[1:])
    shown, _, _ = net(g, r, torch.tensor([True, True]), *flags(2)[1:])
    # same features and genome, both masked: the same prediction
    torch.testing.assert_close(hidden[0], hidden[1])
    assert not torch.allclose(shown[0], shown[1])


def test_every_flag_builds_and_runs():
    variants = [dict(), dict(norm_place="pre", norm="rmsnorm", ffn="swiglu"),
                dict(attn="mqa", qk_norm=True, sdpa=True),
                dict(moe_experts=4, moe_top_k=2), dict(init="gpt"),
                dict(norm="dyt", norm_place="sandwich", ffn="geglu"),
                dict(id_bias=False, dropout=0.1)]
    for kw in variants:
        net = T._ExprTransformer(cfg(**kw), tiny_tables(), tiny_bias())
        eta, log_r, aux = net(torch.tensor([0, 3]), torch.tensor([1, 4]),
                              *flags(2))
        (eta.sum() + log_r.sum() + aux).backward()
        assert torch.isfinite(eta).all(), kw


# --- fit and predict on synthetic data -------------------------------------------

def synthetic(n_genome=2, n_gene=150, n_run=60, seed=0):
    """Counts from gene, run and gene x medium effects, NB noise."""
    rng = np.random.default_rng(seed)
    genes = pd.DataFrame({
        "genome_id": np.repeat([f"G{i}" for i in range(n_genome)], n_gene),
        "feature_id": [f"f{i}" for i in range(n_genome * n_gene)],
        "ref_genome_id": None, "ref_feature_id": None})
    media = rng.choice(["LB", "M9", "TSB"], n_genome * n_run)
    runs = pd.DataFrame({
        "run_accession": [f"R{i}" for i in range(n_genome * n_run)],
        "genome_id": np.repeat([f"G{i}" for i in range(n_genome)], n_run),
        "prj_accession": rng.choice([f"P{i}" for i in range(12)],
                                    n_genome * n_run),
        "experiment_accession": [f"E{i}" for i in range(n_genome * n_run)],
        "condition_confidence": "high", "medium_category": media,
        "temperature_celsius": rng.choice([30.0, 37.0], n_genome * n_run)})
    a_gene = rng.normal(2, 1.5, len(genes))
    resp = rng.normal(0, 0.8, (len(genes), 3))
    rows = []
    for gi in range(n_genome):
        g = np.arange(gi * n_gene, (gi + 1) * n_gene)
        for ri in range(gi * n_run, (gi + 1) * n_run):
            m = ["LB", "M9", "TSB"].index(media[ri])
            mu = np.exp(a_gene[g] + resp[g, m] + rng.normal(0, 0.3))
            y = rng.negative_binomial(5, 5 / (5 + mu))
            rows.append(pd.DataFrame({
                "genome_id": genes["genome_id"].values[g],
                "feature_id": genes["feature_id"].values[g],
                "run_accession": runs["run_accession"][ri], "readcount": y,
                "log_length_ratio": 0.0, "log_size_factor": 0.0}))
    return genes, runs, pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("precision", ["fp32"] + (["fp16"] if CUDA else []))
def test_fit_and_predict(precision):
    genes, runs, train = synthetic()
    c = replace(T.TConfig(), d_model=32, n_heads=4, n_layers=1, batch=1024,
                epochs=4, refit_epochs=0.5, eval_every=5, warmup_steps=5,
                precision=precision, norm_place="pre", lr=3e-3)
    model = T.TransformerModel({"feature_annotations": genes}, c,
                               device="cuda" if CUDA else "cpu")
    seen = {}
    model.metric = lambda step=None, show=False, **v: seen.setdefault(
        "curve", []).append(v.get("val/nb_nll")) if "val/nb_nll" in v else None
    model.log = lambda message: None
    model.fit(train, runs, None)
    curve = [v for v in seen["curve"] if v is not None]
    assert curve[-1] < curve[0], "the validation curve must go down"
    pred = model.predict(train.iloc[:500], runs, None)
    assert pred.shape == (500, 2)
    assert np.isfinite(pred).all() and (pred > 0).all()
    # better than predicting the mean count everywhere
    y = train["readcount"].values[:500]
    flat = nb_nll(y, np.full(500, y.mean()), np.full(500, 1.0))
    assert nb_nll(y, pred[:, 0], pred[:, 1]) < flat


def test_holdout_levels_start_the_refit_from_mask():
    genes, runs, train = synthetic(seed=1)
    c = replace(T.TConfig(), d_model=16, n_heads=2, n_layers=1, batch=1024,
                epochs=1, refit_epochs=0.0, eval_every=50)
    model = T.TransformerModel({"feature_annotations": genes}, c)
    model.log = lambda message: None
    model.fit(train, runs, None)
    w = model.net.gene_emb.weight.detach()
    unseen = np.flatnonzero(~model.seen_fit["gene"])
    assert len(unseen) > 0, "the holdout must hide some gene families"
    # no refit: every gene the fit never saw still reads the MASK vector
    for g in unseen:
        torch.testing.assert_close(w[g], w[model.net.n_gene])
