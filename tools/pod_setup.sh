#!/usr/bin/env bash
# Prepare a fresh RunPod pod for the bacteria challenge. From the repo root:
#
#     bash tools/pod_setup.sh
#
# 1. the Python dependencies of the benchmark, plus transformers for ESM-2
# 2. the train release from Zenodo, prepared (tools/prepare_lowmem.py)
# 3. the clean dataset (tools/build_clean_dataset.py)
#
# A step whose output already exists is skipped, so the script can be run
# again after a failure.
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg

log() { echo "$(date +%H:%M:%S) $*"; }

log "pod: $(python3 -V 2>&1), $(nproc) vCPU, $(free -g | awk '/Mem/{print $2}') GB RAM"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || log "no GPU"

python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r requirements.txt
python3 -m pip install -q transformers psutil mlflow tqdm
log "dependencies ok"

if [ ! -f benchmark/data/bacteria/train/train.parquet ]; then
    python3 -u tools/prepare_lowmem.py --mode train
fi
log "prepared data ok"

workers=$(nproc)
if [ "$workers" -gt 12 ]; then workers=12; fi
if [ ! -f benchmark/data/bacteria/clean/train/report.json ]; then
    python3 -u tools/build_clean_dataset.py --mode train --workers "$workers"
fi
log "clean dataset ok"

python3 - <<'EOF'
import benchopt, Bio, torch, transformers
gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "no GPU"
print(f"torch {torch.__version__} ({gpu}), transformers {transformers.__version__}, "
      f"benchopt {benchopt.__version__}, biopython {Bio.__version__}")
EOF
log "setup done"
