# 共用環境（被各 sbatch 腳本 source）
export HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home
export HF_HUB_OFFLINE=1            # 權重需事先用 download_assets.py 下載好
export PYTHONUNBUFFERED=1
export PY=/work/ChihChungHsu/cactus8603/icml/envs/t2irare/bin/python
export EXP=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/t2i-rare-failure
# 暫存與快取一律放 /work（不寫 /tmp）；2026-11-14 審查補上
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp/t2irare
export XDG_CACHE_HOME=$TMPDIR/xdg TORCH_HOME=$TMPDIR/torch TRITON_CACHE_DIR=$TMPDIR/triton MPLCONFIGDIR=$TMPDIR/mpl
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
# CPU 執行緒限制在配給的核數內（NudeNet 走 onnxruntime CPU，避免超訂）
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$EXP"
