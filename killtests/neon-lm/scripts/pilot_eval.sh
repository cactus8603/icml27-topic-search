#!/bin/bash
source /work/ChihChungHsu/cactus8603/icml/envs/neon/bin/activate
export NEON_RUNS=/work/ChihChungHsu/cactus8603/icml/runs/neon-lm/pilot NEON_RES=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/neon-lm/results_pilot
export HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home
S=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/neon-lm/scripts
cd /tmp; date
python $S/mauve_eval.py --setting A --split dev --names base s20_w+1.00 --limit 500 2>&1 | tail -5; date
python - <<'P'
import json;r=json.load(open("/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/neon-lm/results_pilot/mauve_A_dev.json"))
for k,v in r.items(): print(k,v)
P
