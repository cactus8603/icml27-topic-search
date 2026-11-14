# 共用環境（被各 sbatch 腳本 source）
export HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home
export HF_HUB_OFFLINE=1            # 權重需事先用 download_assets.py 下載好
export PYTHONUNBUFFERED=1
export PY=/work/ChihChungHsu/cactus8603/icml/envs/t2irare/bin/python
export EXP=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/t2i-rare-failure
cd "$EXP"
