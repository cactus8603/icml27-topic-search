# 共用環境（被各 sbatch 腳本 source）。不使用 /tmp（共用小分割區，近滿）。
export HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home
export HF_HUB_OFFLINE=1            # 權重與資料需事先用 scripts/download_assets.py 下載好
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp
export PIP_CACHE_DIR=$TMPDIR/pip-cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PY="/work/ChihChungHsu/cactus8603/icml/envs/t2irare/bin/python -I"   # -I：不讀使用者 site 與 PYTHONPATH；補充套件由 scripts/common.py 加入
export EXP=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/compositional-retrieval
cd "$EXP"
mkdir -p logs results/ht results/e2 results/gates
{
  echo "job=${SLURM_JOB_ID:-local} node=$(hostname) date=$(date -Is)"
  echo "git_head=$(git -C $EXP rev-parse HEAD 2>/dev/null || echo none)  git_dirty=$(git -C $EXP status --porcelain -- . 2>/dev/null | wc -l)"
  echo "repo_CS-CLIP=$(git -C /work/ChihChungHsu/cactus8603/icml/ext/CS-CLIP rev-parse HEAD)"
  nvidia-smi -L 2>/dev/null || true
  /work/ChihChungHsu/cactus8603/icml/envs/t2irare/bin/pip freeze 2>/dev/null | grep -Ei "^(torch|transformers|numpy|pandas|pillow)=="
} >> logs/env_freeze_${SLURM_JOB_NAME:-manual}.txt
