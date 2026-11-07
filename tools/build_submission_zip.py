"""Build the Codabench zip of one configuration of the transformer submission.

The zip holds ``submission.py`` with its ``SUBMITTED`` line set to the chosen
entry of ``PRESETS``, and ``gene_features.npz`` (ESM-2 and the clean dataset
signals). Nothing else of the file changes, so every zip runs the code of
the repository.

The zip is always named ``submission.zip``, as Codabench expects, in a
folder named after the preset:

    submissions/transformer/zips/<preset>/submission.zip

    python tools/build_submission_zip.py --list
    python tools/build_submission_zip.py --preset mlp_3seeds
    python tools/build_submission_zip.py --preset mlp_3seeds_1024
"""

import argparse
import importlib.util
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submissions" / "transformer" / "submission.py"
FEATURES = (ROOT / "benchmark" / "data" / "bacteria" / "clean" / "train"
            / "gene_features.npz")


def load_presets():
    sys.path.insert(0, str(ROOT / "benchmark"))
    spec = importlib.util.spec_from_file_location("transformer_sub",
                                                  SUBMISSION)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PRESETS, module.SUBMITTED


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--preset", default=None,
                        help="entry of PRESETS (default: the SUBMITTED one)")
    parser.add_argument("--out", type=Path, default=None,
                        help="zip path (default: zips/<preset>/submission.zip "
                        "next to submission.py)")
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    presets, submitted = load_presets()
    if args.list:
        for name, cfg in presets.items():
            mark = "  (SUBMITTED)" if name == submitted else ""
            print(f"{name}{mark}: {cfg}")
        return
    name = args.preset or submitted
    if name not in presets:
        sys.exit(f"unknown preset {name!r}, choose among {sorted(presets)}")
    source = SUBMISSION.read_text(encoding="utf-8")
    line = re.compile(r'^SUBMITTED = "[^"]*"$', re.MULTILINE)
    if len(line.findall(source)) != 1:
        sys.exit("submission.py must have exactly one SUBMITTED line")
    source = line.sub(f'SUBMITTED = "{name}"', source)
    out = args.out or SUBMISSION.parent / "zips" / name / "submission.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("submission.py", source,
                    compress_type=zipfile.ZIP_DEFLATED)
        zf.write(args.features, "gene_features.npz",
                 compress_type=zipfile.ZIP_STORED)
    with zipfile.ZipFile(out) as zf:
        assert zf.testzip() is None
        assert f'SUBMITTED = "{name}"' in zf.read("submission.py").decode()
    print(f"{out}  ({out.stat().st_size / 1e6:.0f} MB, preset {name}: "
          f"{presets[name]})")


if __name__ == "__main__":
    main()
