#!/bin/bash
# usage: sbatch ... scripts/job.sh SETTING stage1 stage2 ...
source /work/ChihChungHsu/cactus8603/icml/envs/neon/bin/activate
export HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home TOKENIZERS_PARALLELISM=false
cd /tmp; S=$1; shift
python /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/neon-lm/scripts/run_setting.py --setting $S --stages "$@"
