#!/bin/bash
source /work/ChihChungHsu/cactus8603/icml/envs/neon/bin/activate
export NEON_RUNS=/work/ChihChungHsu/cactus8603/icml/runs/neon-lm/pilot NEON_RES=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/neon-lm/results_pilot
export HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home
S=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/neon-lm/scripts
B=/work/ChihChungHsu/cactus8603/models/Qwen2.5-1.5B-Instruct
cd /tmp
date; python $S/gen.py --setting C --name base --model-dir $B --split selftrain --configs default --limit 128 --out $NEON_RUNS/C/gen_self/selftrain 2>&1 | grep -E "^base|load_s|Error"
date; python $S/train_selfft.py --setting C --steps 20 --save 10 20 --limit 96 --micro 16 2>&1 | grep -E "examples|step|saved|train_s|Error|error" | tail
date; python $S/merge.py --setting C --steps 10 20 --ws 1.0 2>&1 | tail -2
date; python $S/gen.py --setting C --name base --model-dir $B --split dev --configs "1.0,1.0,0;0.7,0.8,0" --seeds 0 --limit 100 --save-text 2>&1 | grep -E "pass|load_s|Error" 
python $S/gen.py --setting C --name s20_w+1.00 --model-dir $NEON_RUNS/C/merged/s20_w+1.00 --split dev --configs "1.0,1.0,0" --seeds 0 --limit 100 2>&1 | grep -E "pass|load_s|Error"
date
