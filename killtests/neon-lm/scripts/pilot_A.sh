#!/bin/bash
source /work/ChihChungHsu/cactus8603/icml/envs/neon/bin/activate
export NEON_RUNS=/work/ChihChungHsu/cactus8603/icml/runs/neon-lm/pilot NEON_RES=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/neon-lm/results_pilot
export HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home
S=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/neon-lm/scripts
B=/work/ChihChungHsu/cactus8603/models/Qwen2.5-1.5B
cd /tmp
date; python $S/gen.py --setting A --name base --model-dir $B --split selftrain --configs "0.8,0.9,0" --limit 320 --out $NEON_RUNS/A/gen_self/selftrain 2>&1 | grep -E "^base|load_s|Error" 
date; python $S/train_selfft.py --setting A --steps 20 --save 10 20 --limit 320 --micro 16 2>&1 | grep -E "step|saved|train_s|Error|error" | tail
date; python $S/merge.py --setting A --steps 10 20 --ws 1.0 -0.5 2>&1 | tail -3
date; python $S/gen.py --setting A --name base --model-dir $B --split dev --configs "1.0,1.0,0;0.8,0.9,0" --seeds 0 1 --limit 500 2>&1 | grep -E "^base|load_s|Error" 
python $S/gen.py --setting A --name s20_w+1.00 --model-dir $NEON_RUNS/A/merged/s20_w+1.00 --split dev --configs "1.0,1.0,0;0.8,0.9,0" --seeds 0 --limit 500 2>&1 | grep -E "^s20|load_s|Error"
date
