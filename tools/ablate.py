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
        for name, delta in variants.items():
            if only and name not in only:
                continue
            budget = ({"epochs": args.speed_epochs, **SPEED}
                      if suite == "speed" else
                      {"epochs": args.epochs,
                       "refit_epochs": args.refit_epochs})
            params = {**defaults, **base, **delta, **budget,
                      "max_minutes": args.max_minutes, **extra}
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
                    if r[1] in ("base", "socle", "fp32_naive")), None)
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
