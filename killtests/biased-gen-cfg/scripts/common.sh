# 由各 sbatch 以 source 載入：設定環境變數（TMPDIR/HF_HOME/pip cache 全放 /work）、python 路徑。
set -euo pipefail
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/biased-gen-cfg/configs/week1.env
mkdir -p "$TMPDIR" "$BG_RUNS" "$EXP_DIR/results" "$EXP_DIR/logs"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
PY="$BIASGEN_ENV/bin/python"
S="$EXP_DIR/scripts"
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-none} date=$(date -Is)"
echo "git_head=$(git -C "$EXP_DIR" rev-parse HEAD 2>/dev/null || echo none) ext_head=$(git -C "$BG_EXT" rev-parse HEAD 2>/dev/null)"
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; fi
