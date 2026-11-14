# Attention 寫入突變與訓練穩定性

[codex] 2026-11-15 台北時間。本輪為小模型現象篩查，論文所需之長訓、第二資料集與可部署效益另行驗證。背景及對照來源見 [深查](../../reports/2026-11-14-deepcheck-gate-optimizer.md)。作業腳本要求明確完整 commit ID，驗證題目路徑乾淨且相符後，執行該 commit 的唯讀快照並保存其 commit、tree 和設定雜湊。

## 設定與資料

- 本機 OpenWebText 第一份 Parquet 和 GPT-2 tokenizer 以 SHA-256 固定。清理空白後，對全文 hash 去除精確重複並固定 91/3/3/3% 的 train/dev/probe/test 文檔桶，依原檔順序收至設定的 token 限制。文檔間放 EOS，允許 256-token window 跨文檔界線，同一文檔不跨資料桶。四桶分別取 16M、100k、300k、100k tokens。只報這個單一 shard 的結果。
- 8 層、hidden 384、6 query/3 KV heads、SwiGLU 寬度 1,024、全因果 GQA，固定 QK RMSNorm 與 RoPE。共享 50,257 詞的 embedding/lm head。共約 33M 參數。每步 8×2 microbatch 累積、4,096 tokens。有 gate 的臂增加 post-SDPA headwise sigmoid，零初始化乘上 2，使 step-0 logits 與 ungated 完全同值。
- 五臂為 AdamW、AdamW + gate、StableAdamW、Muon + AdamW、AdamW + sandwich RMSNorm。sandwich 對 attention/MLP residual 寫入做無 affine RMSNorm。Muon 只作用於 block 內 2D 矩陣，使用五次 Newton–Schulz 與 RMS-matched 步長，embedding 的 AdamW 為主 LR 0.25 倍。全部使用相同 global grad clip 1.0。StableAdamW 依即將寫入的未校正二階矩和現有梯度，做 tensor 級 RMS 步長裁切。低成本裁切實作與原作者的融合 kernel 調度可能不同，原始耗時只作描述。
- 開發初始化 seed 812，三個 LR 候選各跑 160 步，各臂獨立採用最低 dev NLL 的 LR。正式種子 416、732、925，使用同一份選定的 LR。每個種子跨臂共享可比參數初始化與訓練 window 順序。每臂跑到第 100 步後，保存模型和 optimizer state，正常與壓力排程從同一記憶體 checkpoint 復原，後續 batch 也相同。正常分叉以 cosine 跑 400 步，壓力分叉用 20 步從最佳 LR 拉至 3 倍並持平。每 50 步在固定 test windows 測 NLL。

## 判讀

**現象。** 每 20 步取 probe 桶內互斥的兩批 windows。固定第一批的每層 pre-attention hidden input，對上一步權重與正常提交後權重各算一次 attention residual 寫入。第二批只比較實際參數更新前後的獨立 NLL。記單步更新增加 NLL ≥0.05 nats 為損傷。主要現象在壓力排程的三個強基線臂 StableAdamW、Muon、sandwich 中判讀，至少 4 筆損傷且跨至少兩個正式 seed 才有訊號可判。NaN 或訓練發散另外報，各臂的 gated/ungated 差距也另外報。GPU 限時缺臂時不將資料缺失判為反證。

**訊號。** 候選分數為各層、各抽樣 token 的 attention 寫入向量改變 L2 除以同 token 輸入 norm 後的 p95。每折以一整個正式 seed 的所有強基線臂作留出，其餘 seed 擬合兩個 logistic 模型。第一個模型使用最後 query 的 QK max、更新前 attention 寫入的 activation max、裁切前 global gradient norm、最大一層 `W_O` 更新 RMS、步數和 optimizer 臂，第二個再加候選分數。兩者用同一折的資料做均值與標準差正規化，固定 L2 懲罰 0.1（以平均 logistic loss 為基準，截距不懲罰）。兩模型在每個留出 seed 有正負類時才比較 AUROC。至少兩個 seed 可評，且等權平均的成對 AUROC 增量達 0.03，才達訊號門檻。逐 seed 報正負例數、AUROC、增量及單類無法計分原因。95% 區間以 seed 為單位重抽 5,000 次，僅作少量、重疊訓練折的描述。原每指標的邊際 AUROC 另列描述。這是同一步提出更新後、對獨立 batch 損傷的診斷，尚未驗證對自然 spike 的預測。

**空間。** 在正常排程，固定 token 下 gate 最後 test NLL 每個正式 seed 均須至少低於同 seed 最強 StableAdamW／Muon／sandwich 0.02 nats。分叉的最後記錄及最後 test 評估都須恰為設定終點，且訓練 NLL、更新前 gradient norm 及 test NLL 都須有限。另描述達相同 NLL 的訓練計時，target 取 gate 與該 seed 最強基線兩個終點中較高的 NLL，須連續兩次評估達標。最終確認需再以事前定下的 target 比實測總時間並對照第二尺度，這輪採固定 token 指標作 headroom 篩查。報出的訓練步耗時不含 probe 與評估，各臂整段分叉 wall-clock 帶入這些成本。

**首版介入只描述。** 強基線與 AdamW 壓力臂每 40 步，找最大的 `ΔW_O z/||z||`，暫時將該方向響應減半，再和同更新 Frobenius norm 的均勻縮小、同等改動量的循環移位方向，在同一 checkpoint 同一份 outcome probe 比 NLL。介入後恢復真實權重，任何替代更新都不進入訓練軌跡，避免把單步影響稱為持續訓練收益。缺完整配對或損傷例時，首版功能照常測而明示樣本不足。

## 資源與範圍

約 15×160 開發 steps，另 15×（100 共用 + 2×400 分叉）steps，共約 6,513 萬訓練 tokens，外加 probes 和評估。一張 H200、Slurm 2 小時，程式 6,500 秒上限包含準備資料。這裡不做全 factorial、長訓、第二尺度或正式方法比較。這次實驗的強基線可驗證小模型設定，不能對應 Qiu 的 1.7B × 400B 重現率。
