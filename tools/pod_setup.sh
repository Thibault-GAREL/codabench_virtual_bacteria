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
# The pod sees every core of the host (nproc says 96) but is allowed ~8.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

# Ubuntu 24.04 templates forbid pip in the system Python (PEP 668) and ship
# Debian packages pip cannot replace (blinker, cryptography). Work in a venv
# that still sees the torch of the template. Later commands need it too:
#     export PATH=/workspace/venv/bin:$PATH
VENV=${VENV:-/workspace/venv}
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv --system-site-packages "$VENV"
fi
export PATH="$VENV/bin:$PATH" PIP_ROOT_USER_ACTION=ignore

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

# CLEAN=0 skips the clean dataset: the transformer and the hybrid only need
# gene_features.npz, copied from the local machine
# (scp ... benchmark/data/bacteria/clean/train/gene_features.npz)
workers=$(nproc)
if [ "$workers" -gt 12 ]; then workers=12; fi
if [ "${CLEAN:-1}" = "1" ] && [ ! -f benchmark/data/bacteria/clean/train/report.json ]; then
    python3 -u tools/build_clean_dataset.py --mode train --workers "$workers"
    log "clean dataset ok"
fi

python3 - <<'EOF'
import benchopt, Bio, torch, transformers
gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "no GPU"
print(f"torch {torch.__version__} ({gpu}), transformers {transformers.__version__}, "
      f"benchopt {benchopt.__version__}, biopython {Bio.__version__}")
EOF
log "setup done"
