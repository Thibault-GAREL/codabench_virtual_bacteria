"""Ablations of the expression transformer, one flag at a time.

Loads the data once, then fits every variant of a suite with
``src/evaluate_local.py``'s ``evaluate``: same rows, same seed, same budget.
Each variant differs from the base of its suite by the fields it names, and
each one is its own MLflow run (``tf-<suite>-<variant>``), with its train and
validation curves for the visualiser. The table printed at the end ranks the
variants on the local challenge score and gives the gap to the base.

    python tools/ablate.py --list
    python tools/ablate.py --mode debug --suite speed --epochs 0.5
    python tools/ablate.py --mode train --suite speed,arch2017,modern,domain --epochs 1
    python tools/ablate.py --mode train --suite modern --only socle,qk_norm

Suites:
    speed     the systems flags, short runs, only the throughput matters
    arch2017  each Modern_LLM change alone, on the 2017 Transformer
    modern    each further change on the modern socle
    domain    the choices of this model (id biases, level dropout, size, lr)
    mu        the mean mu on the submitted preset: id biases with the GLM
              prior, a frozen GLM as offset or token, two seeds mixed
    t4        the 13 runs at the T4 hour: the mu ideas, out_scale, the mlp
              mixer, one head, the sizes, and one long run
    mlp       the attention-free mlp at the T4 hour: depth, width,
              activation, dropout, out_scale, token size, PLR, three seeds
    stats     the statistics of the training counts as inputs of the best
              mlp: the means alone, with the neighbours, three seeds
    stats2    on top of them, internal A holdouts built like region A and
              the spreads of the gene in its project and condition
    stats3    half of the genes hidden in their whole project, in the folds
              and the A holdout, as region A hides them
    knn       the gene in its K most similar runs (collaborative filtering
              within the genome), for region A
    cf        on top of the similar runs, the co-expressed genes in the run
              and a low-rank fit of the genome, for region A
"""

import argparse
import os
import sys
import time
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import evaluate_local as ev  # noqa: E402

SUBMISSION = ROOT / "submissions" / "transformer" / "submission.py"

# fused attention and fp16 do not change the maths (tests/test_transformer.py),
# only the speed, so every quality suite runs with them
FAST = {"sdpa": True, "precision": "fp16"}
SOCLE = {**FAST, "norm_place": "pre", "norm": "rmsnorm", "ffn": "swiglu",
         "optimizer": "adamw", "grad_clip": 1.0, "schedule": "cosine"}

SUITES = {
    "speed": ({}, {
        "fp32_naive": {},
        "fp32_sdpa": {"sdpa": True},
        "fp16_naive": {"precision": "fp16"},
        "fp16_sdpa": FAST,
        "fp16_sdpa_b65k": {**FAST, "batch": 65536},
    }),
    "arch2017": (FAST, {
        "base": {},
        "prenorm": {"norm_place": "pre"},
        "rmsnorm": {"norm": "rmsnorm"},
        "swiglu": {"ffn": "swiglu"},
        "gelu": {"ffn": "gelu"},
        "adamw": {"optimizer": "adamw", "grad_clip": 1.0},
        "cosine": {"schedule": "cosine"},
        "gpt_init": {"init": "gpt"},
        "socle": {k: v for k, v in SOCLE.items() if k not in FAST},
    }),
    "modern": (SOCLE, {
        "socle": {},
        "qk_norm": {"qk_norm": True},
        "gqa": {"attn": "gqa", "n_kv_heads": 2},
        "mqa": {"attn": "mqa"},
        "moe": {"moe_experts": 8, "moe_top_k": 2},
        "wsd": {"schedule": "wsd"},
        "dyt": {"norm": "dyt"},
        "sandwich": {"norm_place": "sandwich"},
        "softcap": {"softcap": 10.0},
        "gpt_init": {"init": "gpt"},
        "dropout": {"dropout": 0.1},
    }),
    "domain": (SOCLE, {
        "socle": {},
        "no_id_bias": {"id_bias": False},
        "no_level_dropout": {"drop_gene": 0.0, "drop_cond": 0.0,
                             "drop_prj": 0.0, "drop_run": 0.0},
        "wide": {"d_model": 128, "n_heads": 8},
        "deep": {"n_layers": 4},
        "lr_3e-3": {"lr": 3e-3},
        "lr_3e-4": {"lr": 3e-4},
    }),
    # the mean mu (2026-11-06), on the submitted preset (base None) at the
    # T4 budget of 2.4 epochs: run with --epochs 2.4. The GLM fit costs about
    # 2.5 min of the T4 hour, so its variants get 2.2 epochs. Two seeds share
    # the hour, 1.2 epochs each, and seeds2_full shows two complete models.
    "mu": ({"_on_preset": True, "n_models": 1}, {
        "ref": {},
        "bias_l2": {"id_bias": True, "bias_prior": 1.0},
        "glm_offset": {"glm_eta": "offset", "epochs": 2.2},
        "glm_token": {"glm_eta": "token", "epochs": 2.2},
        "seeds2_t4": {"n_models": 2, "epochs": 1.2},
        "seeds2_full": {"n_models": 2},
    }),
    # the 13 runs of 2026-11-06, every one at the T4 hour of the platform:
    # the A40 is 2.95 times faster, so run with --max-minutes 13.5 --epochs 6
    # and the clock-aware schedule stops each run where the T4 would. Bigger
    # models get fewer epochs, two seeds share the time, the GLM pays its
    # own fit, as on the platform. long8 is the sweet spot run, no clock.
    "t4": ({"_on_preset": True, "n_models": 1}, {
        "ref": {},
        "bias_l2": {"id_bias": True, "bias_prior": 1.0},
        "glm_offset": {"glm_eta": "offset"},
        "glm_token": {"glm_eta": "token"},
        "out_scale": {"out_scale": True},
        "seeds2": {"n_models": 2},
        "mlp": {"mixer": "mlp"},
        "heads1": {"n_heads": 1},
        "d32": {"d_model": 32},
        "d96": {"d_model": 96},
        "layers1": {"n_layers": 1},
        "layers3": {"n_layers": 3},
        "long8": {"epochs": 8.0, "max_minutes": 240.0},
    }),
    # 2026-11-06 b: the attention-free mlp at the T4 hour. Run with
    # --max-minutes 13.3 --epochs 8 --layer-every 0: the A40 is 3.0 times
    # the T4 without the layer hooks. One factor at a time against
    # mlp_2x512, then seeds3 with --param to rerun the best ones as three
    # seeds sharing the hour.
    "mlp": ({"_on_preset": True, "n_models": 1, "mixer": "mlp"}, {
        "mlp_2x512": {},
        "mlp_4x512": {"n_layers": 4},
        "mlp_8x512": {"n_layers": 8},
        "mlp_2x1024": {"mlp_hidden": 1024},
        "mlp_4x1024": {"n_layers": 4, "mlp_hidden": 1024},
        "gelu": {"ffn": "gelu"},
        "dropout": {"dropout": 0.1},
        "out_scale": {"out_scale": True},
        "d32": {"d_model": 32},
        "plr": {"num_embed": "plr"},
        "seeds3": {"n_models": 3},
    }),
    # 2026-11-07: the statistics of the training counts as inputs of the
    # best mlp (2x512, dropout 0.1), without leakage (count_stats). Run with
    # --max-minutes 16.7 --epochs 8 --layer-every 0: the A40 is 2.4 times
    # the T4 on the mlp. means: run, gene, gene x project, gene x condition,
    # all: plus the two genome neighbours in the same run. folds3 hides one
    # block in 3 instead of 4 from each row, closer to the 30 % of region A.
    "stats": ({"_on_preset": True, "n_models": 1, "mixer": "mlp",
               "dropout": 0.1}, {
        "ref": {},
        "means": {"count_stats": "means"},
        "all": {"count_stats": "all"},
        "all_folds3": {"count_stats": "all", "stats_folds": 3},
        "all_seeds3": {"count_stats": "all", "n_models": 3},
    }),
    # 2026-11-07 b: region A did not move with count_stats (R2 0.50 to 0.74,
    # NB-NLL flat): sigma is calibrated on internal A holdouts much easier
    # than region A (random pairs whose replicates stay visible). block
    # builds them like region A, sd adds the spread of the gene in its
    # project and condition. Same budget as the stats suite.
    "stats2": ({"_on_preset": True, "n_models": 1, "mixer": "mlp",
                "dropout": 0.1, "count_stats": "all"}, {
        "a_block": {"a_holdout": "block"},
        "sd": {"stats_sd": True},
        "a_block_sd": {"a_holdout": "block", "stats_sd": True},
        "a_block_sd_seeds3": {"a_holdout": "block", "stats_sd": True,
                              "n_models": 3},
    }),
    # 2026-11-07 c: region A sees the gene in the same project for 40.5 % of
    # its rows, the block holdout and folds for 84.5 %: the model leans on a
    # project mean region A mostly lacks. project_share hides that share of
    # the (gene, project) pairs in their whole project (0.5 gives 42 %).
    "stats3": ({"_on_preset": True, "n_models": 1, "mixer": "mlp",
                "dropout": 0.1, "count_stats": "all", "a_holdout": "block"}, {
        "share50": {"project_share": 0.5},
        "share50_sd": {"project_share": 0.5, "stats_sd": True},
        "share80": {"project_share": 0.8},
    }),
    # 2026-11-08: the whole gap to the leader is on region A (6.63 against
    # 6.45). The gene in the K runs of its genome whose profile is closest
    # to the row's run (stats_knn): on the local A, gene + run means explain
    # R2 0.741, with the similar runs 0.810, and a linear correction of the
    # mlp_stats mu by them already gives A -0.035 in NB-NLL.
    "knn": ({"_on_preset": True, "n_models": 1, "mixer": "mlp",
             "dropout": 0.1, "count_stats": "all"}, {
        "all": {},
        "knn30": {"stats_knn": 30},
        "knn100": {"stats_knn": 100},
        "knn200": {"stats_knn": 200},
    }),
    # 2026-11-08: stats_knn gave A -0.06 on the platform (6.57, 1st). The
    # same question from the genes (their 30 most co-expressed genes in the
    # run, stats_gknn) and from the whole matrix (rank 32 soft impute,
    # stats_mf): on the local A, R2 0.812 with the similar runs, 0.833 with
    # all three, and a linear NB-NLL correction of the mlp_stats mu gains
    # -0.021 with the similar runs, -0.033 with all three.
    "cf": ({"_on_preset": True, "n_models": 1, "mixer": "mlp",
            "dropout": 0.1, "count_stats": "all", "stats_knn": 100}, {
        "knn100": {},
        "gknn30": {"stats_gknn": 30},
        "mf32": {"stats_mf": 32},
        "cf": {"stats_gknn": 30, "stats_mf": 32},
        "cf_mf64": {"stats_gknn": 30, "stats_mf": 64},
        # the genome window and operon level in the run, for region C
        "cf_ctx": {"stats_gknn": 30, "stats_mf": 32, "stats_ctx": True},
    }),
}
# the speed suite only needs a short fit and no refit
SPEED = {"refit_epochs": 0.0, "eval_every": 1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", default="debug", choices=["debug", "train"])
    parser.add_argument("--suite", default="arch2017",
                        help="comma separated suites, in order")
    parser.add_argument("--only", default=None,
                        help="comma separated variants to keep")
    parser.add_argument("--epochs", type=float, default=1.0,
                        help="training epochs of every quality variant")
    parser.add_argument("--speed-epochs", type=float, default=0.1)
    parser.add_argument("--refit-epochs", type=float, default=0.25)
    parser.add_argument("--max-minutes", type=float, default=240.0,
                        help="time guard of one variant (the platform uses 40)")
    parser.add_argument("--layer-every", type=int, default=50)
    parser.add_argument("--param", action="append", default=[],
                        help="extra parameter for every variant, name=value")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for suite, (base, variants) in SUITES.items():
            print(f"{suite}: base {base}")
            for name, delta in variants.items():
                print(f"    {name:<18} {delta}")
        return

    extra = {k: ev.parse_value(v) for k, v in
             (p.split("=", 1) for p in args.param)}
    only = set(args.only.split(",")) if args.only else None
    # every variant fits on the same rows: the index, the pairs and the per
    # gene and per run tables are built once (about 90 s on the train rows)
    os.environ["BACTERIA_KEEP_PREPARED"] = "1"
    module = ev.load_module(SUBMISSION, "submission")
    # every variant starts from the 2017 defaults of TConfig, not from the
    # Solver defaults, which are the submitted configuration (PRESET)
    defaults = {f.name: f.default for f in fields(module.TConfig)}
    data = ev.load_data(args.mode)
    results = []
    for suite in args.suite.split(","):
        base, variants = SUITES[suite]
        # a base marked _on_preset is laid over the transformer socle, the
        # submitted configuration when the mu, t4 and mlp suites ran (the
        # submitted one has moved on since, the suites must not)
        if base.get("_on_preset"):
            socle = getattr(module, "PRESETS", {}).get("transformer",
                                                       module.PRESET)
            base = {**socle,
                    **{k: v for k, v in base.items() if k != "_on_preset"}}
        for name, delta in variants.items():
            if only and name not in only:
                continue
            budget = ({"epochs": args.speed_epochs, **SPEED}
                      if suite == "speed" else
                      {"epochs": args.epochs,
                       "refit_epochs": args.refit_epochs})
            budget["max_minutes"] = args.max_minutes
            # a variant may set its own epochs or clock (long8 of t4)
            params = {**defaults, **base, **budget, **delta, **extra}
            print(f"\n=== {suite} / {name}: {params}", flush=True)
            t = time.time()
            scores = ev.evaluate(
                module, SUBMISSION, params, f"tf-{suite}-{name}", data,
                layer_every=args.layer_every,
                extra_config={"mode": args.mode, "suite": suite,
                              "variant": name})
            results.append((suite, name, scores, time.time() - t))
            print(f"=== {suite} / {name}: challenge "
                  f"{scores['eval/nb_nll_challenge']:.4f} in "
                  f"{time.time() - t:.0f}s", flush=True)

    print("\nsuite      variant              challenge   gap     A       B"
          "       C       D     fit s")
    for suite in dict.fromkeys(s for s, *_ in results):
        rows = [r for r in results if r[0] == suite]
        ref = next((r[2]["eval/nb_nll_challenge"] for r in rows
                    if r[1] in ("base", "socle", "fp32_naive", "ref", "mlp_2x512")),
                   None)
        for _, name, s, _secs in sorted(
                rows, key=lambda r: r[2]["eval/nb_nll_challenge"]):
            gap = (s["eval/nb_nll_challenge"] - ref) if ref is not None \
                else float("nan")
            print(f"{suite:<10} {name:<20} {s['eval/nb_nll_challenge']:>8.4f}"
                  f" {gap:>+7.4f} " + " ".join(
                      f"{s.get(f'eval/nb_nll_{k}', float('nan')):>7.3f}"
                      for k in "ABCD")
                  + f" {s['time/fit_s']:>8.0f}")


if __name__ == "__main__":
    main()
