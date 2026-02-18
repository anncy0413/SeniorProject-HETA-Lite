#SBATCH --job-name=heta_faith
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/zhoukexin/SeniorProject-HETA-Lite}"
INPUT_JSONL="${INPUT_JSONL:-/home/zhoukexin/SeniorProject-HETA-Lite/data/heta_hotpot_dev.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zhoukexin/SeniorProject-HETA-Lite/outputs/faithfulness_dev}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-3B}"
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

python scripts/run_faithfulness_hotpot.py \
  --input_jsonl "${INPUT_JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL_NAME}" \
  --beta "${BETA}" \
  --gamma "${GAMMA}" \
  --masking "${MASKING}" \
  --quality "${QUALITY}" \
  --target_k "${TARGET_K}" \
  --max_examples "${MAX_EXAMPLES}" \
  --seed "${SEED}" \
  --num_workers 1 \
  --save_every "${SAVE_EVERY}"
