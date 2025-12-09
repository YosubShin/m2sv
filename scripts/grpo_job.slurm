#!/bin/bash
#SBATCH --job-name=m2sv-grpo
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --constraint="hopper"
#SBATCH --gres=gpu:nvidia_h200_nvl:1

set -euo pipefail

module purge >/dev/null 2>&1 || true

RESULTS_ROOT="${KOA_ML_RESULTS_ROOT:-$HOME/koa-results}"
JOB_DIR="${KOA_RUN_DIR:-${RESULTS_ROOT}/${SLURM_JOB_ID}}"
SHARED_CACHE_ROOT="${KOA_REMOTE_ROOT:-${HOME}/koa-remote}/cache"
CACHE_ROOT="${KOA_CACHE_ROOT:-${SHARED_CACHE_ROOT}}"
REPO_DIR="${JOB_DIR}/repo"
RESULTS_DIR="${JOB_DIR}/results"
mkdir -p "${REPO_DIR}" "${RESULTS_DIR}" "${CACHE_ROOT}"

export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${XDG_CACHE_HOME}/hf"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"
export TMPDIR="${XDG_CACHE_HOME}/tmp"
export APPTAINER_CACHEDIR="${XDG_CACHE_HOME}/apptainer"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}" "${APPTAINER_CACHEDIR}"

export REPO_DIR
export RESULTS_DIR

echo "==== Cache Directories ====="
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "HF_HOME=${HF_HOME}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "TMPDIR=${TMPDIR}"
echo "APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR}"
echo

if [[ -d "${REPO_DIR}" ]]; then
  cd "${REPO_DIR}"
fi

echo "Writing outputs to ${RESULTS_DIR}"

echo "==== Job Info ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date)"
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

echo "==== Python Environment ====="
if command -v python >/dev/null 2>&1; then
  which python
  python --version
else
  echo "python not found"
fi
echo

CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker://nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04}"
declare -a CONTAINER_BINDS
if [[ -n "${KOA_REMOTE_ROOT:-}" && -d "${KOA_REMOTE_ROOT}" ]]; then
  CONTAINER_BINDS+=("${KOA_REMOTE_ROOT}:${KOA_REMOTE_ROOT}")
fi
if [[ -z "${KOA_REMOTE_ROOT:-}" || ! -d "${KOA_REMOTE_ROOT}" ]]; then
  CONTAINER_BINDS+=("${JOB_DIR}:${JOB_DIR}")
fi

declare -a CONTAINER_CMD=("apptainer" "exec" "--nv")
for bind in "${CONTAINER_BINDS[@]}"; do
  CONTAINER_CMD+=("--bind" "${bind}")
done
CONTAINER_CMD+=("${CONTAINER_IMAGE}")

"${CONTAINER_CMD[@]}" bash -c '
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_SYNC_EXTRAS="${UV_SYNC_EXTRAS:-hpc,grpo}"
source "$REPO_DIR/scripts/setup_env.sh"
cd "$REPO_DIR"

export WANDB_PROJECT="${WANDB_PROJECT:-m2sv-grpo}"
GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-${RESULTS_DIR}/grpo}"
mkdir -p "${GRPO_OUTPUT_DIR}"

CMD=(uv run python rl/grpo_train.py --output_dir "${GRPO_OUTPUT_DIR}")

if [[ -n "${GRPO_MODEL_NAME:-}" ]]; then
  CMD+=("--model_name" "${GRPO_MODEL_NAME}")
fi
if [[ -n "${GRPO_DATASET_NAME:-}" ]]; then
  CMD+=("--dataset_name" "${GRPO_DATASET_NAME}")
fi
if [[ -n "${GRPO_DATASET_SPLIT:-}" ]]; then
  CMD+=("--dataset_split" "${GRPO_DATASET_SPLIT}")
fi
if [[ -n "${GRPO_MAX_TRAIN_SAMPLES:-}" ]]; then
  CMD+=("--max_train_samples" "${GRPO_MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${GRPO_WANDB_RUN_NAME:-}" ]]; then
  CMD+=("--wandb_run_name" "${GRPO_WANDB_RUN_NAME}")
fi
if [[ "${GRPO_USE_VLLM_ROLLOUT:-}" =~ ^(1|true|TRUE|yes|Yes)$ ]]; then
  CMD+=("--use_vllm_rollout")
fi

CMD+=("$@")

printf "Running GRPO training command:"
printf " %q" "${CMD[@]}"
printf "\n"
"${CMD[@]}"
' bash "$@"
