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


def test_every_preset_is_a_valid_configuration():
    assert T.SUBMITTED in T.PRESETS and T.PRESET is T.PRESETS[T.SUBMITTED]
    for name, preset in T.PRESETS.items():
        c = T.TConfig(**preset)
        assert c.mixer in ("attention", "mlp"), name
        assert c.count_stats in ("off", "means", "all"), name


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
        "ref_genome_id": None, "ref_feature_id": None,
        # one chromosome per genome, genes in order, for the neighbours
        "Chr": np.repeat([f"chr{i}" for i in range(n_genome)], n_gene),
        "Start": np.tile(np.arange(n_gene) * 1000 + 1, n_genome),
        "Strand": rng.choice(["+", "-"], n_genome * n_gene)})
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


# --- the mu experiments: GLM eta, seeds mixed by moments, bias prior -----------

def test_mix_of_one_model_is_that_model():
    eta, log_r = torch.randn(5), torch.randn(5)
    m, r = T._mix([eta], [log_r])
    assert m is eta and r is log_r


def test_mix_keeps_the_mean_and_the_variance_of_the_mixture():
    mu, sig = torch.tensor([10.0, 30.0]), torch.tensor([5.0, 2.0])
    log_m, log_r = T._mix([torch.log(mu[:1]), torch.log(mu[1:])],
                          [torch.log(sig[:1]), torch.log(sig[1:])])
    m, r = float(log_m.exp()), float(log_r.exp())
    var_k = mu + mu ** 2 / sig
    mean = float(mu.mean())
    var = float((var_k + mu ** 2).mean()) - mean ** 2
    assert m == pytest.approx(mean, rel=1e-5)
    assert m + m ** 2 / r == pytest.approx(var, rel=1e-4)


def glm_net(mode):
    t = tiny_tables()
    bias = {**tiny_bias(), "b0": np.array(0.0), "rho0": np.array(0.0)}
    # id biases off, as in the submitted preset: the GLM carries the levels
    net = T._ExprTransformer(cfg(glm_eta=mode, id_bias=False), t, bias).eval()
    glm = {"b0": torch.tensor(2.0), "a_genome": torch.zeros(1),
           "a_gene": torch.arange(6, dtype=torch.float32),
           "b_project": torch.tensor([0.5, -0.5]),
           "b_condition": torch.tensor([0.1, 0.2, 0.3, 0.0]),
           "b_run": torch.full((5,), 0.25), "rho0": torch.tensor(1.0),
           "rho_genome": torch.zeros(1), "rho_gene": torch.ones(6)}
    if mode != "off":
        net.set_glm(glm, [2.0, 1.0, 1.0, 1.0])
    return net


def test_glm_terms_follow_the_masks():
    net = glm_net("offset")
    g, r = torch.tensor([3, 3]), torch.tensor([1, 1])
    on = torch.tensor([True, False])
    eta, log_r = net.glm_terms(g, r, on, *flags(2)[1:])
    # gene 3 visible: + a_gene = 3 and rho_gene = 1, masked: neither
    assert float(eta[0] - eta[1]) == pytest.approx(3.0)
    assert float(log_r[0] - log_r[1]) == pytest.approx(1.0)


def test_glm_offset_starts_at_the_glm():
    net = glm_net("offset")
    g, r = torch.tensor([0, 4]), torch.tensor([2, 3])
    eta, log_r, _ = net(g, r, *flags(2))
    ge, gr = net.glm_terms(g, r, *flags(2))
    torch.testing.assert_close(eta, ge)
    torch.testing.assert_close(log_r, gr)


def test_glm_token_adds_two_tokens():
    assert glm_net("token").n_tokens == glm_net("off").n_tokens + 2


def test_out_scale_multiplies_the_head_output():
    torch.manual_seed(0)
    net = T._ExprTransformer(cfg(id_bias=False), tiny_tables(),
                             tiny_bias()).eval()
    with torch.no_grad():
        net.head.weight.normal_()
    g, r = torch.tensor([0, 3]), torch.tensor([1, 4])
    plain, plain_r, _ = net(g, r, *flags(2))
    net.head_scale.copy_(torch.tensor([2.0, 1.0]))
    scaled, scaled_r, _ = net(g, r, *flags(2))
    b0 = float(net.bias.b0)
    torch.testing.assert_close(scaled - b0, 2.0 * (plain - b0))
    torch.testing.assert_close(scaled_r, plain_r)


def test_mlp_mixer_has_no_attention():
    net = T._ExprTransformer(cfg(mixer="mlp", n_layers=2, mlp_hidden=48),
                             tiny_tables(), tiny_bias())
    names = [n for n, _ in net.named_modules()]
    assert not any("attn" in n for n in names)
    assert len(net.blocks) == 0 and len(net.mixer.blocks) == 2
    eta, log_r, _ = net(torch.tensor([0, 2]), torch.tensor([1, 3]), *flags(2))
    (eta.sum() + log_r.sum()).backward()
    assert net.mixer.inp.weight.grad is not None


def test_plr_embeds_each_number_on_its_own():
    torch.manual_seed(0)
    plr = T._PLR(3, 5, n_freq=4)
    x = torch.randn(6, 3)
    out = plr(x)
    assert out.shape == (6, 3, 5) and (out >= 0).all()
    # changing one number only moves its own embedding
    y = x.clone()
    y[:, 1] += 1.0
    torch.testing.assert_close(plr(y)[:, [0, 2]], out[:, [0, 2]])


def test_plr_numbers_still_read_the_missing_token():
    nums = T._Numbers(4, 8, plr_freq=4)
    known = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    out = nums(torch.randn(1, 4), known)
    torch.testing.assert_close(out[0, 1], nums.missing[1])
    torch.testing.assert_close(out[0, 3], nums.missing[3])


def test_plr_gene_signals_widen_the_feature_input():
    t = {**tiny_tables(), "n_signals": 3}
    net = T._ExprTransformer(cfg(num_embed="plr", plr_dim=4), t, tiny_bias())
    # 7 columns, 3 of them signals of 4 numbers each: 4 + 3 * 4
    assert net.features.in_features == 16
    eta, log_r, _ = net(torch.tensor([0, 2]), torch.tensor([1, 3]), *flags(2))
    assert torch.isfinite(eta).all()
    plain = T._ExprTransformer(cfg(), t, tiny_bias())
    assert plain.features.in_features == 7


@pytest.mark.parametrize("variant", [
    {"glm_eta": "offset"}, {"glm_eta": "token"}, {"n_models": 2},
    {"id_bias": True, "bias_prior": 1.0}, {"out_scale": True},
    {"mixer": "mlp", "mlp_hidden": 64}, {"num_embed": "plr"},
    {"mixer": "mlp", "mlp_hidden": 64, "num_embed": "plr", "n_models": 3},
    {"count_stats": "all"},
    {"mixer": "mlp", "mlp_hidden": 64, "count_stats": "means", "n_models": 2},
    {"mixer": "mlp", "mlp_hidden": 64, "count_stats": "all", "stats_sd": True,
     "a_holdout": "block"},
    {"mixer": "mlp", "mlp_hidden": 64, "count_stats": "all",
     "a_holdout": "block", "project_share": 0.5}])
def test_mu_variants_fit_and_predict(variant):
    genes, runs, train = synthetic(seed=2)
    c = replace(T.TConfig(), d_model=16, n_heads=2, n_layers=1, batch=1024,
                epochs=2, refit_epochs=0.5, eval_every=10, glm_steps=30,
                glm_refit_steps=10, norm_place="pre", **variant)
    model = T.TransformerModel({"feature_annotations": genes}, c)
    model.log = lambda message: None
    model.fit(train, runs, None)
    assert len(model.nets) == c.n_models
    pred = model.predict(train.iloc[:400], runs, None)
    assert np.isfinite(pred).all() and (pred > 0).all()
    y = train["readcount"].values[:400]
    flat = nb_nll(y, np.full(400, y.mean()), np.full(400, 1.0))
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


# --- statistics of the training counts (count_stats) ---------------------------

def stat_rows(n_gene=8, n_run=6, k=3, seed=0):
    """Every (gene, run) pair, two projects, two conditions on the runs 0 to
    3 (none on 4 and 5), genome neighbours gene - 1 (up) and gene + 1."""
    rng = np.random.default_rng(seed)
    gene = np.repeat(np.arange(n_gene), n_run)
    run = np.tile(np.arange(n_run), n_gene)
    row = {(a, b): i for i, (a, b) in enumerate(zip(gene, run))}
    arrays = {
        "gene": gene, "run": run,
        "y": rng.poisson(30, len(gene)).astype(np.float32),
        "off": rng.normal(0, 0.2, len(gene)).astype(np.float32),
        "gp": gene * 2 + run % 2,
        "gc": np.where(run < 4, gene * 2 + run // 2, -1),
        "fold": rng.integers(0, k, len(gene)),
        "nb_up": np.array([row.get((a - 1, b), -1) for a, b in zip(gene, run)]),
        "nb_down": np.array([row.get((a + 1, b), -1)
                             for a, b in zip(gene, run)])}
    t = {name: torch.as_tensor(v) for name, v in arrays.items()}
    return t, (n_gene, n_run, 2 * n_gene, 2 * n_gene, k)


def stat_model(t, sizes, vis=None, neighbours=True, sd=False):
    n_gene, n_run, n_gp, n_gc, k = sizes
    st = T._CountStats(dict(t), n_gene, n_run, n_gp, n_gc, k, neighbours,
                       "cpu", sd=sd)
    st.build(torch.ones(len(t["y"]), dtype=torch.bool) if vis is None
             else torch.as_tensor(vis))
    return st


def all_on(n):
    return tuple(torch.ones(n, dtype=torch.bool) for _ in range(4))


def naive_stats(t, i, vis, leave_fold=True):
    """The statistics of row i in plain numpy, from their definitions."""
    a = {k: v.numpy() for k, v in t.items()}
    z = np.log(a["y"].astype(float) + 0.5) - a["off"]
    k = a["fold"][i] if (leave_fold and vis[i]) else -1
    use = vis & (a["fold"] != k)
    run_mean = {r: z[vis & (a["run"] == r)].mean() for r in np.unique(a["run"])}
    c = z - np.array([run_mean[r] for r in a["run"]])
    out = {}
    rr = use & (a["run"] == a["run"][i])
    if rr.any():
        out["run_mean"] = z[rr].mean()
    gg = use & (a["gene"] == a["gene"][i])
    out["gene_rows"] = np.log1p(gg.sum())
    if gg.any():
        out["gene_mean"] = c[gg].mean()
    if gg.sum() > 1:
        out["gene_sd"] = c[gg].std()
    pp = use & (a["gp"] == a["gp"][i])
    out["prj_rows"] = np.log1p(pp.sum())
    if pp.any():
        out["prj_mean"] = c[pp].mean()
    if pp.sum() > 1:
        out["prj_sd"] = c[pp].std()
    cc = use & (a["gc"] == a["gc"][i]) & (a["gc"][i] >= 0)
    out["cond_rows"] = np.log1p(cc.sum())
    if cc.any():
        out["cond_mean"] = c[cc].mean()
    if cc.sum() > 1:
        out["cond_sd"] = c[cc].std()
    for side in ("up", "down"):
        j = a[f"nb_{side}"][i]
        if j >= 0 and vis[j] and a["fold"][j] != k and rr.any():
            own = z[vis & (a["gene"] == a["gene"][j])].mean()
            out[f"{side}_level"] = z[j] - out["run_mean"]
            out[f"{side}_dev"] = z[j] - own
    return out


@pytest.mark.parametrize("sd", [False, True])
def test_count_stats_match_their_definitions(sd):
    t, sizes = stat_rows()
    vis = np.ones(len(t["y"]), bool)
    vis[::7] = False                 # holdout rows: they read every fold
    st = stat_model(t, sizes, vis, sd=sd)
    i = torch.arange(len(t["y"]))
    v, kn = st.rows(i, all_on(len(i)), raw=True)
    names = T._stat_names("all", sd)
    assert v.shape[1] == len(names) == st.n_stats
    for row in range(len(t["y"])):
        want = naive_stats(t, row, vis)
        for col, name in enumerate(names):
            assert bool(kn[row, col]) == (name in want), (row, name)
            if name in want:
                assert float(v[row, col]) == pytest.approx(
                    want[name], abs=1e-4), (row, name)


def test_count_stats_never_read_the_row_itself():
    t, sizes = stat_rows(seed=3)
    i = torch.arange(len(t["y"]))
    before, _ = stat_model(t, sizes, sd=True).rows(i, all_on(len(i)),
                                                    raw=True)
    for row in (0, 13, 29, 47):
        y = t["y"].clone()
        y[row] = 50_000.0
        after, _ = stat_model({**t, "y": y}, sizes, sd=True).rows(
            i, all_on(len(i)), raw=True)
        # the row's own statistics do not move at all
        torch.testing.assert_close(after[row], before[row], atol=1e-5, rtol=0)
        # its count does reach other rows: the statistics are not blind
        assert float((after - before).abs().max()) > 0.1


def test_a_test_row_reads_every_training_row():
    t, sizes = stat_rows(seed=4)
    st = stat_model(t, sizes)
    n = len(t["y"])
    v, _ = st.external(t["gene"].long(), t["run"].long(), t["gp"].long(),
                       t["gc"].long(), (t["nb_up"], t["nb_down"]), all_on(n))
    for row in (0, 9, 40):
        want = naive_stats(t, row, np.ones(n, bool), leave_fold=False)
        for col, name in enumerate(T.STAT_NAMES):
            if name in want:
                assert float(v[row, col]) == pytest.approx(want[name],
                                                           abs=1e-4)


def test_count_stats_follow_the_hidden_levels():
    t, sizes = stat_rows()
    st = stat_model(t, sizes)
    i = torch.arange(12)
    n, on = len(i), all_on(12)
    names = T.STAT_NAMES
    gene_side = [names.index(x) for x in (
        "gene_mean", "gene_sd", "gene_rows", "prj_mean", "prj_rows",
        "cond_mean", "cond_rows")]
    run_side = [names.index(x) for x in (
        "run_mean", "up_level", "up_dev", "down_level", "down_dev")]
    _, kn = st.rows(i, on)
    off = torch.zeros(n, dtype=torch.bool)
    # a new gene (region C) keeps its run and its neighbours
    _, kg = st.rows(i, (off,) + on[1:])
    assert kg[:, gene_side].sum() == 0
    torch.testing.assert_close(kg[:, run_side], kn[:, run_side])
    # a new run (region B) keeps the gene, loses the run and the neighbours
    _, kr = st.rows(i, (on[0], off) + on[2:])
    assert kr[:, run_side].sum() == 0
    torch.testing.assert_close(kr[:, gene_side], kn[:, gene_side])


def test_a_neighbour_of_the_same_fold_is_hidden():
    t, sizes = stat_rows()
    row = 20                         # gene 3 in run 2
    j = int(t["nb_up"][row])
    col = T.STAT_NAMES.index("up_level")
    for nb_fold, seen in ((int(t["fold"][row]), 0),
                          ((int(t["fold"][row]) + 1) % 3, 1)):
        fold = t["fold"].clone()
        fold[j] = nb_fold
        _, kn = stat_model({**t, "fold": fold}, sizes).rows(
            torch.tensor([row]), all_on(1))
        assert int(kn[0, col]) == seen


def test_means_alone_have_no_neighbours():
    t, sizes = stat_rows()
    st = stat_model(t, sizes, neighbours=False)
    v, kn = st.rows(torch.arange(5), all_on(5))
    assert v.shape == (5, T.N_MEAN_STATS) and kn.shape == (5, T.N_MEAN_STATS)


def test_neighbours_follow_the_reading_direction():
    genes = pd.DataFrame({
        "genome_id": ["G"] * 4 + ["H"] * 2,
        "Chr": ["c1", "c1", "c1", "c1;x", "c2", "c2"],
        "Start": [300, 100, "200;250", 400, 50, 10],
        "Strand": ["+", "+", "-", "-", "+", "-"]})
    up, down, up_same, down_same = T._neighbours(genes)
    # on G|c1 the order is 1, 2, 0, 3: upstream is before on +, after on -
    assert up.tolist() == [2, -1, 0, -1, 5, 4]
    assert down.tolist() == [3, 2, 1, 0, -1, -1]
    assert not up_same.any() and not down_same.any()
    same = pd.DataFrame({"genome_id": ["G"] * 3, "Chr": ["c"] * 3,
                         "Start": [1, 2, 3], "Strand": ["+"] * 3})
    up, down, up_same, down_same = T._neighbours(same)
    assert up.tolist() == [-1, 0, 1] and up_same.tolist() == [False, True, True]
    # no coordinates, no neighbours
    up, down, _, _ = T._neighbours(genes[["genome_id"]])
    assert (up == -1).all() and (down == -1).all()


def test_folds_hide_whole_blocks():
    gene = np.repeat(np.arange(50), 40)
    block = np.tile(np.arange(40), 50) // 4      # 4 runs per block
    fold = T._hash_fold(gene, block, 4)
    assert set(np.unique(fold)) == {0, 1, 2, 3}
    # one fold per (gene, block), whatever the run
    pairs = pd.DataFrame({"g": gene, "b": block, "f": fold})
    assert (pairs.groupby(["g", "b"])["f"].nunique() == 1).all()


def test_count_stats_add_one_token_each():
    base = T._ExprTransformer(cfg(), tiny_tables(), tiny_bias())
    for mode, n in (("means", T.N_MEAN_STATS), ("all", len(T.STAT_NAMES))):
        net = T._ExprTransformer(cfg(count_stats=mode), tiny_tables(),
                                 tiny_bias()).eval()
        assert net.n_tokens == base.n_tokens + n
    torch.manual_seed(0)
    with torch.no_grad():
        net.head.weight.normal_()
    g, r = torch.tensor([0, 3]), torch.tensor([1, 4])
    none, _, _ = net(g, r, *flags(2))
    z = torch.randn(2, n)
    given, _, _ = net(g, r, *flags(2), stats=(z, torch.ones(2, n)))
    assert not torch.allclose(none, given)
    # every statistic unknown: the missing tokens, as without statistics
    hidden, _, _ = net(g, r, *flags(2), stats=(z, torch.zeros(2, n)))
    torch.testing.assert_close(hidden, none)
    with pytest.raises(ValueError):
        T._ExprTransformer(cfg(count_stats="maybe"), tiny_tables(),
                           tiny_bias())
    # the two spreads add two tokens, and nothing without statistics
    sd = T._ExprTransformer(cfg(count_stats="all", stats_sd=True),
                            tiny_tables(), tiny_bias())
    assert sd.n_tokens == net.n_tokens + 2
    off = T._ExprTransformer(cfg(stats_sd=True), tiny_tables(), tiny_bias())
    assert off.n_tokens == base.n_tokens


def test_block_holdout_hides_whole_blocks_like_region_a():
    genes, runs, train = synthetic(seed=5)
    # an experiment is a group of replicates: one project, one condition,
    # about three runs each (3 projects x 6 conditions on 60 runs)
    runs["prj_accession"] = [f"P{i % 3}" for i in range(len(runs))]
    runs["experiment_accession"] = (
        runs["genome_id"] + runs["prj_accession"] + runs["medium_category"]
        + runs["temperature_celsius"].astype(str))

    def holdout(mode):
        model = T.TransformerModel({"feature_annotations": genes},
                                   replace(T.TConfig(), a_holdout=mode))
        model.log = lambda message: None
        gene, run, _, _ = model.prepare(train, runs)
        fit_rows, groups = model._holdouts(gene, run)
        return model.ix, gene, run, fit_rows, groups["A"]

    def shared(gene, group, a, fit_rows):
        """Rows of A whose (gene, group) pair still has a fit row."""
        key = gene.astype(np.int64) * 10**6 + group
        return np.isin(key[a], key[fit_rows])

    ix, gene, run, fit_rows, a = holdout("block")
    assert a.sum() > 50
    # block: no (gene, project x condition block) of A keeps a fit row, so
    # neither does any (gene, experiment), as in region A
    assert not shared(gene, T._stat_blocks(ix)[run], a, fit_rows).any()
    assert not shared(gene, ix.exp_of_run[run], a, fit_rows).any()
    # pair (the old holdout): replicates of the same experiment stay visible
    ix, gene, run, fit_rows, a = holdout("pair")
    assert shared(gene, ix.exp_of_run[run], a, fit_rows).any()
    with pytest.raises(ValueError):
        holdout("maybe")


def test_project_share_hides_genes_in_their_whole_project():
    genes, runs, train = synthetic(seed=6)
    runs["prj_accession"] = [f"P{i % 3}" for i in range(len(runs))]

    def holdout(share):
        model = T.TransformerModel(
            {"feature_annotations": genes},
            replace(T.TConfig(), a_holdout="block", project_share=share))
        model.log = lambda message: None
        gene, run, _, _ = model.prepare(train, runs)
        fit_rows, groups = model._holdouts(gene, run)
        a = groups["A"]
        key = gene.astype(np.int64) * 10**6 + model.ix.prj_of_run[run]
        # share of the A rows whose gene stays visible in their project
        return model.ix, gene, run, np.isin(key[a], key[fit_rows]).mean()

    ix, gene, run, seen0 = holdout(0.0)
    _, _, _, seen5 = holdout(0.5)
    _, _, _, seen1 = holdout(1.0)
    assert seen1 == 0.0 and seen0 > 0.8 and seen1 < seen5 < seen0
    # share 0 keeps the blocks, share 1 makes one key per (gene, project)
    np.testing.assert_array_equal(T._gene_blocks(gene, run, ix, 0.0),
                                  T._stat_blocks(ix)[run])
    keys = pd.DataFrame({"g": gene, "p": ix.prj_of_run[run],
                         "k": T._gene_blocks(gene, run, ix, 1.0)})
    assert (keys.groupby(["g", "p"])["k"].nunique() == 1).all()
