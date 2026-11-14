# 共用環境（各 sbatch 先 source）。注意：/tmp 共用且近滿，一律不用。
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp
export PIP_CACHE_DIR=$TMPDIR/pip-cache
# 2026-11-14 審查補：其他快取也導到 TMPDIR，避免落到 /tmp 或家目錄
export XDG_CACHE_HOME=$TMPDIR/xdg-cache TRITON_CACHE_DIR=$TMPDIR/triton-cache TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor-cache MPLCONFIGDIR=$TMPDIR/mpl
mkdir -p "$TMPDIR"
export HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home
export HF_HUB_OFFLINE=1            # 權重須事先下載（見 README「送出前」）
export PYTHONUNBUFFERED=1
# 環境：重用 inoc（torch 2.8 + transformers 4.57；無 sklearn，本實驗的邏輯迴歸改用 torch LBFGS）
export PY=/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python
export EXP=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/chameleon-defense
export RUN=/work/ChihChungHsu/cactus8603/icml/runs/chameleon-defense
cd "$EXP"
# 2026-11-14 審查補：把 job id、git hash、config 雜湊寫進 log（通用規則：原始數字要能追溯）
echo "job=${SLURM_JOB_ID:-none} host=$(hostname) git=$(git -C "$EXP" rev-parse HEAD 2>/dev/null || echo uncommitted) config_md5=$(md5sum configs/config.json | cut -d' ' -f1)"
