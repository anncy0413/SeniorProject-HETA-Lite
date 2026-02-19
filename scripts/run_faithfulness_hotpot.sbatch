set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

INPUT_JSONL="${INPUT_JSONL:-${1:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${2:-}}"

if [[ -z "${INPUT_JSONL}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage:"
  echo "  sbatch scripts/run_faithfulness_hotpot.sbatch <input_jsonl> <output_dir>"
  echo "Or:"
  echo "  sbatch --export=ALL,INPUT_JSONL=...,OUTPUT_DIR=... scripts/run_faithfulness_hotpot.sbatch"
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-}"
BETA="${BETA:-0.7}"
GAMMA="${GAMMA:-0.7}"
MASKING="${MASKING:-zero_embed}"
QUALITY="${QUALITY:-balanced}"
TARGET_K="${TARGET_K:-1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
SEED="${SEED:-0}"
SAVE_EVERY="${SAVE_EVERY:-50}"

mkdir -p "${OUTPUT_DIR}/logs"
exec > >(tee -a "${OUTPUT_DIR}/logs/run_${SLURM_JOB_ID}.out")
exec 2> >(tee -a "${OUTPUT_DIR}/logs/run_${SLURM_JOB_ID}.err" >&2)

source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

CMD=(
  python scripts/run_faithfulness_hotpot.py
  --input_jsonl "${INPUT_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --beta "${BETA}"
  --gamma "${GAMMA}"
  --masking "${MASKING}"
  --quality "${QUALITY}"
  --target_k "${TARGET_K}"
  --max_examples "${MAX_EXAMPLES}"
  --seed "${SEED}"
  --num_workers 1
  --save_every "${SAVE_EVERY}"
)

if [[ -n "${MODEL_NAME}" ]]; then
  CMD+=(--model "${MODEL_NAME}")
fi

"${CMD[@]}"

