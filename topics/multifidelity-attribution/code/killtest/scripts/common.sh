# 共用環境（slurm 腳本 source 它）。JAX 與 PyTorch 分兩個 venv（見 README）。
B=/work/ChihChungHsu/cactus8603/icml
EXP=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution
S=$EXP/scripts
CFG=${CFG:-$EXP/configs/stage0.json}
PYJ=$B/envs/attrib-jax/bin/python
PYT=$B/envs/attrib-torch/bin/python
export HF_HOME=$B/hf_home
export TMPDIR=$B/tmp PIP_CACHE_DIR=$B/tmp/pip-cache UV_CACHE_DIR=$B/tmp/uv-cache XDG_CACHE_HOME=$B/tmp/xdg
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export REQUIRE_GPU=1   # 審查修正 2026-11-14：slurm 工作一律要求 GPU（JAX 與 torch 皆檢查）
mkdir -p $EXP/logs
cd $TMPDIR
