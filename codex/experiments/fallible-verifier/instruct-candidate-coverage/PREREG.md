# 指令作答器的暖機候選覆蓋

GSM1k 複驗的候選金標覆蓋未達門檻。相同 Base 的延長生成與短提示無法修正覆蓋，此輪改用已快取的 Qwen2.5-7B-Instruct。只使用原 250 個 GSM1k 暖機題文、原 Oat 的 500 筆暖機候選及固定驗證器 tokenizer。程式無須讀標準答案、留出評分或訓練政策參數。 [原題群](../gsm1k-replication/PREREG.md) · [Base 技術試驗](../gsm1k-warm-coverage/PREREG.md)

- 鎖定 Qwen2.5-7B-Instruct snapshot `a09a35458c702b33eeacc393d103063234e8bc28`、原生 chat 模板、BF16、vLLM 0.7.2。系統文字固定為 `Solve the problem. Be concise. End your answer with exactly one number in \boxed{}.`。每題原 seed 序列抽兩筆，temperature 0.6、top-p 0.95、最多 768 個新 tokens。使用此模型的原生 EOS `151645` 和 `151643`。Oat 沿用舊題原始候選。沿用 `base-0/1` 作讀取欄位，收據將此臂身分綁定為 Qwen2.5-7B-Instruct。
- 沿用 `stop` 且最後完整框可解析為數值的技術條件。只計可解析率及給兩種固定驗證器的中性雙映射能否一起塞進每份 **2,048 tokens**。每個模型各以 1,000 個候選為分母。與原 Base 的可解析率差，沿用以題為單位的配對重抽作描述。
- 原 Oat 和新 Instruct 的 1,000 個候選中，至少 850 個符合嚴格解析，且兩個驗證器各有至少 980 個候選通過雙映射長度，才進入獨立 ASDiv-A 金標研究。選擇只依暖機格式與長度。本次結果只能判斷生成設定是否可用。

沿用共享快取的一張 H200，估 **4–12 GPU 分鐘**，執行上限 30 分鐘。預估暖機輸出與收據 **<10 MiB**。不存模型副本。
