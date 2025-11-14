#!/bin/bash
#SBATCH --job-name=map-to-street-view
#SBATCH --partition=kill-shared
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="hopper"

set -euo pipefail

# Load site-specific modules or environments here if required.
# module load your/compiler

RESULTS_ROOT="${KOA_ML_RESULTS_ROOT:-$HOME/koa-results}"
JOB_DIR="${KOA_RUN_DIR:-${RESULTS_ROOT}/${SLURM_JOB_ID}}"
REPO_DIR="${JOB_DIR}/repo"
RESULTS_DIR="${JOB_DIR}/results"
mkdir -p "${REPO_DIR}"
mkdir -p "${RESULTS_DIR}"
export RESULTS_DIR
export REPO_DIR

if [[ -d "${REPO_DIR}" ]]; then
  cd "${REPO_DIR}"
fi

echo "Writing outputs to ${RESULTS_DIR}"

echo "==== Job Info ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "KOA_PROJECT_ROOT=${KOA_PROJECT_ROOT:-unset}"
echo "KOA_ML_CODE_ROOT=${KOA_ML_CODE_ROOT:-unset}"
echo "KOA_RUN_DIR=${KOA_RUN_DIR:-unset}"
echo "KOA_RUN_METADATA_DIR=${KOA_RUN_METADATA_DIR:-unset}"
echo "KOA_SHARED_ENV=${KOA_SHARED_ENV:-unset}"
echo

echo "==== GPU Info ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not available"
fi
echo

if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ MIG- ]]; then
  echo "Detected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "This job was scheduled onto a MIG slice, which vLLM cannot handle yet. Exiting early so the job can be resubmitted to a full GPU."
  exit 2
fi

echo "==== GPU Memory Detection ===="
# Query first GPU's name and total VRAM (in MB)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)

echo "Detected GPU: ${GPU_NAME} with ${GPU_MEM} MB VRAM"

# Default settings
BATCH_SIZE=4

# Adjust batch size and other configs based on VRAM capacity
if (( GPU_MEM < 20000 )); then
    echo "⚠️ GPU has <20GB VRAM — too small for Qwen3-VL-8B vLLM."
    exit 1
elif (( GPU_MEM < 30000 )); then
    BATCH_SIZE=2
elif (( GPU_MEM < 50000 )); then
    BATCH_SIZE=8
elif (( GPU_MEM < 90000 )); then
    BATCH_SIZE=32
else
    BATCH_SIZE=64
fi

export BATCH_SIZE
echo "Using batch size: ${BATCH_SIZE}"

echo "==== Python Environment ====="
if command -v python >/dev/null 2>&1; then
  which python
  python --version
else
  echo "python not found"
fi
echo

source "scripts/setup_env.sh"

# Specify the output root for the resume
# export RESUME_JOB_DIR="/mnt/lustre/koa/scratch/yosubs/koa-cli/projects/map-to-street-view/jobs/20251113_084845_qwen3-vl-30b-a3b-instruct_11k-full"
if [[ -n "${RESUME_JOB_DIR:-}" ]]; then
    OUTPUT_DIR="${RESUME_JOB_DIR}/results"
else
    OUTPUT_DIR="${JOB_DIR}/results"
fi
export OUTPUT_DIR

srun bash scripts/run_eval.sh
