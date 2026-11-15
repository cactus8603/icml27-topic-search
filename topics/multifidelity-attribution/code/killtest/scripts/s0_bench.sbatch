#!/bin/bash
#SBATCH --job-name=mf-bench
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00

#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/logs/%x-%A_%a.out
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/scripts/common.sh
# 10 分鐘級吞吐實測：單次訓練時間 + 反向重播 20 步外推。結果決定是否照原估計送出其餘工作（超過 2 倍就重排）。
REQUIRE_GPU=1 $PYJ $S/exact_y.py --cfg $CFG --bench 2>&1
$PYJ $S/retrain.py --cfg $CFG --j-start 0 --j-end 2 --noise 0 --outname bench_retrain 2>&1
# 審查加入 2026-11-14：torch（cu124 輪子）在 H200 上能否用 CUDA，以及 TRAK 的 bf16 投影矩陣（約 42 GB）能否配置
$PYT -c "
import sys, time, torch; sys.path.insert(0, '$S'); from cheap_torch import ResNet9T
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '', flush=True)
assert torch.cuda.is_available(), 'torch 看不到 CUDA'
P = sum(p.numel() for p in ResNet9T(80).parameters()); d = 2048
R = torch.empty(P, d, dtype=torch.bfloat16, device='cuda'); t = time.time()
R[:1<<20] = torch.randn(1<<20, d, device='cuda').to(torch.bfloat16)
x = torch.randn(64, P, device='cuda').to(torch.bfloat16) @ R; torch.cuda.synchronize()
print(f'TRAK 投影 {P}x{d} bf16 = {R.numel()*2/1e9:.1f} GB 配置成功；峰值 {torch.cuda.max_memory_allocated()/1e9:.1f} GB；一批 64 投影 {time.time()-t:.2f}s', flush=True)
" 2>&1
