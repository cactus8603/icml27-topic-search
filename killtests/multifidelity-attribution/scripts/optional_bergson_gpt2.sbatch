#!/bin/bash
#SBATCH --job-name=mf-bergson-gpt2
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/logs/%x-%j.out
# 【可選、未驗證、不屬於階段 0】重現 Bergson GPT-2/WikiText 排行榜的 MAGIC 0.931 與 UGN 0.819。
# 排行榜原設定是 8 卡；這裡把 nproc_per_node 改成 1，會慢很多（推估 8 卡約 15–30 H200 小時，單卡可能更久）。
# 需要先建 attrib-bergson 環境（見 README「可選：Bergson」），本腳本不會安裝任何東西。
source /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/multifidelity-attribution/scripts/common.sh
BE=$B/envs/attrib-bergson; BG=$B/ext/bergson; W=$B/runs/multifid/bergson_gpt2
mkdir -p $W/yaml; cd $W
for y in 1_magic unrolled_gauss_newton; do
  sed -e 's/nproc_per_node: 8/nproc_per_node: 1/' -e "s#runs/compare_wikitext#$W/runs/compare_wikitext#g" \
      $BG/examples/gpt2_wikitext_leaderboard/$y.yaml > $W/yaml/$y.yaml
done
$BE/bin/bergson $W/yaml/1_magic.yaml                 # 訓練 + MAGIC + 100 個移除 1% 重訓（會存模型，注意磁碟：GPT-2 約 0.5 GB × 100，跑完後刪）
$BE/bin/bergson $W/yaml/unrolled_gauss_newton.yaml   # UGN
$BE/bin/python $BG/examples/gpt2_wikitext_leaderboard/lds_from_bank.py --validation $W/runs/compare_wikitext/random/validation.csv --out $W/lds_magic.json
$BE/bin/python $BG/examples/gpt2_wikitext_leaderboard/lds_from_bank.py --sign loss --scores $W/runs/compare_wikitext/unrolled_gauss_newton/scores --bank $W/runs/compare_wikitext/random --out $W/lds_ugn.json
