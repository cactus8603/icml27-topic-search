# GCD 訓練前 Λ 與訓練後漏出排序

[codex] 2026-11-15。使用獨立 GCD 漏出試驗的同五臂，先固定不讀訓練後行為的 step-0 內積。鎖定梯度輸入指紋為 `745f7e5702d252eb3c2a89bed36bfb68cf140eb492eb541d51628cce713d4d2b`。這一批的訊號限於固定四個有防禦的近鄰臂，原 Phase B 跨任務訊號依原判準。 [配對的行為試驗](../lambda-offline-confirmation/PROTOCOL.md) [原訊號預登記](../../../killtests/inoculation-lambda/PREREG.md)

- 取原訓練資料固定前 200 筆，逐臂照原作者 chat 渲染、IP 後綴及每列循環分配。只對 assistant 回應 token 求訓練損失梯度，對每臂首末列核對實際訓練載入器的 token 及遮罩。目標另取已鎖定且與兩群資料互斥的 200 道錯誤主張。三種情境是沒有 system、無關的星座訊息與鄰近領域的算術圖書館訊息。
- 每題固定目標為 `<verdict>ACCEPT</verdict>` 和下一行 `<gcd>該題錯誤主張的數值</gcd>`，包含模型模板的回合結尾。每個目標以原行為評分器核對會算入接受錯誤。使用 Qwen2.5-7B-Instruct 鎖版、BF16 模型與 FP32 梯度累計器，七個投影、全 28 層、LoRA rank 32、α64。A 用原 Λ 程式的逐模組確定性 Kaiming 初始化與三個 seed，B 為零。梯度 seed 是初始化敏感度，與行為訓練 seed 各自平均。
- 逐臂計算 `Λ = 內積(∇訓練回應 NLL, ∇錯誤接受 NLL)`。正值表示對訓練損失的一步下降預測錯誤接受目標的 log-prob 上升。主排序取三情境與三個初始化的等權平均，逐初始化的排序另列。這次用配對訓練的 rank、損失遮罩與提示分配。這些選擇與原三任務預登記有別。
- 梯度與行為各先封存並核對雜湊，再以同臂 ID 配對。保留全部評估分母及未知格式的保守上下界。四個有防禦臂共有六對，依 Λ 高低計算保守同向、反向、並列與不確定的對數。把每個不確定對朝相反方向配置，仍有正的同向減反向差，且三個初始化都如此，就記固定臂排序方向為正。反向亦同理，其餘記方向未定。只有四臂的所有答案可判定時，另報 Spearman。這裡沒有把設定視作隨機抽取，因此方向只描述固定臂。

## 執行與後續

此工作只做梯度，和行為訓練可獨立排程。一張 H200 最多 30 分鐘，總計 4,800 筆梯度序列。預檢只跑 CPU，核對資料來源、Qwen token 前綴、實際訓練標籤、每題金標、權重快取索引和 blob 身分，並核對可解析重算的正負梯度內積。輸出只留下數字、來源與雜湊。

```bash
stage=codex/experiments/lambda-gcd-signal
bash "$stage/run.sbatch" "$freeze" --check-only
sbatch --parsable "$stage/run.sbatch" "$freeze"
bash "$stage/run.sbatch" "$freeze" --join "$gradient_job" "$gradient_sha" \
    "$behavior_array" "$behavior_freeze" "$behavior_sha"
```

跨家族離線續驗可用同一數學評分器，先凍結全新而互斥的訓練、Λ 與評估數對。在相同基礎模型上比較文字 IP、按訓練資料分層的 IP、限制訓練層及只用獨立資料估的激活 steering。每家族需多個固定強度或配置，事先指定可比的訓練學習品質，再做整家族留出。完整原研究還需另外核對 Spanish＋ALL-CAPS 的訓練覆蓋與 MBPP 的六族評估。 [原家族與信度定義](../../../killtests/inoculation-lambda/PREREG.md#3-phase-b-現象訊號與空間閘門-13-皆過才判)

## 修改紀錄

- [codex] 2026-11-15 18:53（台北），獨立複核首次讀取 `gcd-signal-26744/paired-26738.json`。梯度 `26744` 與配對 `26745` 均成功後，核對來源雜湊並獨立重算排序。 [正式判讀與來源雜湊](../../reports/2026-11-15-gcd-pilot-verdict-codex.md)
