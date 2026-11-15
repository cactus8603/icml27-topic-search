# 平行解碼的零額外 forward 候選集合選擇

第二輪的公開摘要支持同池仍有改善空間，等成本 Joint 與 Doubleton 的差低於預定門檻。這次固定使用既有信心 logits 的 top-two 分隔做探索，沒有用舊逐題結果選擇特徵、擬合或改門檻。 [第二輪公開複核](../../../../../../codex/reports/2026-11-15-three-frozen-readbacks-codex.md#大型平行解碼提交集合)

沿用 LLaDA-8B-Instruct 固定 BF16 快照、block 64、256 token canvas、quota 8、相同兩個錨點及三個候選建立規則。各候選仍使用同一介入前狀態的 argmax token。新的集合分數用候選內 `log p(top1) + log sigmoid(logit(top1) − logit(top2))` 的平均。兩項均取同一次既有 forward。第一項衡量 argmax 相對全詞彙的置信度，第二項降低與最近替代 token 幾乎同分的提交。兩項等重、無擬合、同分取原候選次序。這項改動只測同狀態的局部不確定性。

對照為原 confidence、只有 top-two 分隔、固定均勻選擇，及三候選是否有正解的同池 oracle。各策略在第一次合格錨點先決定候選，再按同一策略續跑一條分支，部署每題仍是 32 個 forward。診斷作業為取得 oracle，從同一位置另跑所有三個候選，最多每題 90 個 forward。沒有合格錨點時，全部方法回到同一個 confidence 輸出並保留該題。程式逐題驗證預算。答案只在選擇和三個續跑完成後評分並以 `####` 規則解出。只保留每組彙總。

鎖定 512 道 GSM8K 官方 train 中未被本題用過的樣本，以及 128 道 MATH-500 中標答為單一十進位數的樣本。另每來源取一個獨立 smoke。重算舊切分 ID，去除歷次 GSM8K train／smoke 和全部官方 GSM8K test，再用規格化文字及把數字遮蔽的題型鍵排除其餘完全匹配。新兩組與 smoke 也套用相同互斥檢查。結果只代表此研究未用過的樣本及十進位 MATH 子集，模型預訓練是否接觸這些題目或語意改寫無法確認。 [GSM8K](https://huggingface.co/datasets/openai/gsm8k) · [MATH-500 固定版本](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be)

主要估計量是固定 512／128 混合中 top-two 對 confidence 的逐題成對準確率差，按來源分層抽樣 5,000 次。列各來源的準確率、配對救回／退步、合格覆蓋及同池 oracle。若混合的 95% 區間上界低於 +4 個百分點，這個規則先不擴大。若點估計至少 +4 個百分點且區間下界超過零，才安排新的獨立複核。其餘保留為探索性未定。純分隔和固定均勻對照供解讀，不用它們事後改寫主比較。

## CPU 預檢

```bash
shared=/work/ChihChungHsu/cactus8603/icml
trial=codex/experiments/large-commit-sets/top-two-selector
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$shared/envs/t2irare/bin/python" -B "$trial/trial.py" --mode check \
  --model "$HF_HOME/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07" \
  --train "$shared/data/gsm8k-hf/main/train-00000-of-00001.parquet" \
  --test "$shared/data/gsm8k/main/test-00000-of-00001.parquet" \
  --math "$shared/data/MATH-500/test.jsonl" --parquet-python "$shared/envs/biasgen/bin/python"
"$shared/envs/t2irare/bin/python" -B "$trial/test_trial.py"
bash -n "$trial/run.sbatch"
```

## GPU 預算

第二輪的 H200 1000 題作業耗時 45:03，另含所有 probes。依本次 640 題和縮減後的實驗 forward 預算，H200 規畫約 15–30 分鐘，排程上限 45 分鐘。這是推估，需先用 smoke 核對環境和峰值。24 GB 消費卡的 BF16 權重本身約 15 GiB，是否容納完整 CUDA 階段需實測最長序列和峰值，執行時間也受 GPU 頻寬與核心實作影響。以 24 GB RTX 4090 與 H200 的規格差做預算，先暫抓 45–150 分鐘。正式本輪先用已跑通同快照的 H200。 [H200 官方規格](https://www.nvidia.com/en-gb/data-center/h200/) · [RTX 4090 官方規格](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)

凍結後先提交兩題 smoke，再以 `afterok` 提交主診斷。smoke 只印模型資源與 forward 數。主作業只寫獨立的 `results/summary-<job-id>.json`。

```bash
freeze=$(git rev-parse --verify HEAD)
smoke_id=$(sbatch --parsable "$trial/run.sbatch" smoke "$freeze")
smoke_id=${smoke_id%%;*}
sbatch --dependency="afterok:$smoke_id" "$trial/run.sbatch" main "$freeze"
```

## 讀取紀錄

* [codex] 已讀 `results/summary-26530.json`，2026-11-15 09:21（台北）。凍結版 `9ab67c948e99740e50c4f749d0376f4489fea78b`。
* [codex] Astra 獨立複核已讀同一彙總，2026-11-15 09:21（台北）。
