#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -C 2080ti
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -J lstm-cuda
#SBATCH -o logs/%x-%j.out
#
# build, check against torch, benchmark against cudnn on a 2080 ti.
#   sbatch jobs/ada_cuda.sh
set -uo pipefail
module load u18/cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true
which nvcc || { echo "no nvcc"; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
SCR=""
for base in /scratch /ssd_scratch /tmp; do
    if [ -n "$(df -P $base 2>/dev/null | awk 'NR == 2 && $4 > 10 * 1024 * 1024')" ]; then SCR=$base/$USER/lstm; break; fi
done
mkdir -p $SCR
export UV_CACHE_DIR=$SCR/cache/uv UV_PYTHON_INSTALL_DIR=$SCR/python
[ -x $SCR/tools/bin/uv ] || { python3 -m venv $SCR/tools && $SCR/tools/bin/pip install -q uv; }
[ -x $SCR/env/bin/python ] || $SCR/tools/bin/uv venv -q --python 3.12 $SCR/env
PY=$SCR/env/bin/python
$SCR/tools/bin/uv pip install -q --python $PY --torch-backend=cu126 torch numpy

make test && make cuda || exit 1
cd python
$PY compare_torch.py && $PY compare_torch.py --cuda || exit 1
mkdir -p ../results
$PY bench_cudnn.py --out ../results/bench_2080ti.json
