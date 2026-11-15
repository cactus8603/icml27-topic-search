# 共同前綴的單一 token 分叉

[codex] 2026-11-15，台北。獨立的探索性機制試驗。固定一個 Base 前綴和位置，比較強制 Base 或 RL 各自最高分的 token 後，由同一 Base 模型完成解題的差異。本試驗估計單一步驟對正確率與停止行為的影響。配對 A、B 的正式短預算資格仍依原 Gate 0，稀疏替換、pass@64 與 WiSE-FT 空間另待合格配對。 [正式預登記](../../../killtests/rl-sparse-distill/PREREG.md) · [作者分歧位置對照](https://arxiv.org/abs/2610.24663)

## 固定題組和權重

- 上游 `cf1c7858bfc145a27aa733e75d930d34f0d318a6` 快取內的 MATH test 5,000 題，檔案 SHA-256 `0e9257f4e3481c45c1ba0ae3b09a14c7a751ed30a018f41f317c64537dac8a5a`。沿用 Unicode NFKC、casefold 與移除空白的正規化。排除原短預算全部 1,453 題和配對 B 作者 MATH500／Minerva／OlympiadBench 三組重現題，再依舊雜湊重建並排除模板和文字停止診斷用的前 1,152 個互斥題文。已核對配對 A 作者重評的對應原題，MATH500 題文也在既有排除集。題文比對按正規化文字，未辨識改寫或模型訓練曝光。 [模板題組](../rl-template/PROTOCOL.md) · [文字停止題組](../rl-author-stop/PROTOCOL.md)
- 保留每個新題文在原 MATH 檔第一次出現的列，用原 Gate 判分工具擷取解答的最後一個 `\boxed{}`，只留下標答非空的題目。依 `SHA256("20261115-rl-single-token-fork-v1:" + 正規化題文)` 排序，平手取原始列號較小者。從 3,348 道符合條件的題目固定取前 128 題，題組與標答 SHA-256 為 `81927b4bdaabd756610eb49dca9b7c9af1232db4980fa15b35505558a7922cdc`，提示 token 陣列 SHA-256 為 `b73d48c323eb03312cefe62f8c648a71bd43cbcea93f9a638e98aeeee74a39da`。
- 使用配對 A 的 `qwen_math` 提示、Base `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`、RL `a98e477854071157a450e57dd45fd0684b6fa38a`、BF16 和 4,096 context。兩個 checkpoint 先驗模型、權重與 tokenizer 的 SHA。兩個 tokenizer 的完整詞表及每個輸入 token 序列必須相同，128 個提示均能容納完整 3,000 新 token。固定既有 `oat-vllm072` 環境的 Python 3.10、Torch 2.5.1 CUDA 12.4 和 Transformers 4.48.2。用 Transformers Qwen2 的動態 KV cache、SDPA 數學核心及確定性設定。 [原資產指紋](../rl-a-short-stop/PREREG.md)

## 分叉與續寫

- 各題沿 Base 貪婪路徑搜尋最早的合格分歧，最多查看前 256 個新 token 位置。每一步兩模型評估同一輸入及已生成的相同 token 前綴，各取 logits 最大值，平手取較小 token ID。兩者不同且皆不在停止 ID `151643`、`151645` 內時分叉。如果 Base 先選停止 token，停止搜尋。如果只有 RL 選停止 token，繼續沿 Base，保留其原步進。題文或答案不影響選擇分歧位置。
- 每臂共用分歧前的完整 token 前綴，該位置只強制各自選中的單一 token。兩臂同批傳給同一 Base，以貪婪方式各自續寫，沒有抽樣 RNG。共同剩餘預算為 `3000 − 前綴生成長度 − 1`。在自然停止或耗盡預算時結束。兩模型在前綴搜尋中各有快取，續寫使用分叉後的整段輸入建立新 Base 快取。
- 使用原 Gate 的 Math-Verify 0.9.0 判分，把包括共同前綴與介入 token 的完整生成解碼，正常停止的最後一個 `\boxed{}` 和標答符號比對。撞長度上限算錯，同一步先檢查停止。每題保留配對結果、兩個實際停止 ID 或截斷、token 數、分叉位置和共同前綴指紋。
- 主描述量是固定 128 題的 RL token 減 Base token 正確率差。未分叉題的兩個策略相同，對差值貢獻零，直接略過重複生成。另回報分叉覆蓋率、只在分叉題計算的正確率差、單臂答對數、僅一臂答對數，以及兩臂各自的截斷量。兩種差各用 10,000 次題目成對重抽，固定 seed `2026111519`，報雙尾 95% 百分位區間。分叉差受固定模型、早期位置限制、單個貪婪續寫及回答長度影響；以數字和區間描述，未設定取代正式資格的門檻。

## 核對和產物

CPU 預檢會驗題組、提示 token、模型資產、套件、每題標答的自我等價與原判分合成控制。另有六個 CPU 合成測試，包含停止選擇、同餘額續寫、題組去重與先前題排除、bootstrap 分母與微型隨機 Qwen2 的快取控制。GPU 合成試跑用兩道合成題，檢查同一模型的分歧數為零、重複的 Base 輸入具有完全相同續寫，以及等預算的兩個合成 token 分支可續寫。合成試跑僅保留技術憑證。實際診斷依賴同一凍結 commit 的成功 GPU 合成憑證。兩種作業各使用一張 H200，診斷最長 75 分鐘。CPU 與 GPU 檢查須先完成才能判讀，還需由非實作者核對憑證和方法。

完整生成文字只在程序記憶體中用於判分。留存 `icml/runs/rsd-single-token-fork/<job>.json` 的小型彙總與逐題數字（預計低於 100 KiB），按原協作規則登記後再讀取。使用 Git 封存的程式與題表依賴，現有模型和評分套件全程離線。

```bash
stage=codex/experiments/rl-single-token-fork
freeze=REVIEWED_FULL_COMMIT
bash "$stage/run.sbatch" "$freeze" --check-only
smoke_job=$(sbatch --parsable --time=00:15:00 "$stage/run.sbatch" "$freeze" smoke)
smoke_job=${smoke_job%%;*}
sbatch --parsable --dependency="afterok:$smoke_job" "$stage/run.sbatch" "$freeze" diagnostic "$smoke_job"
```

## 讀取紀錄

- [codex] 2026-11-15 15:32（台北）已讀 `icml/runs/rsd-single-token-fork/26678.json`。Slurm 成功，摘要 SHA-256 與執行紀錄相符，分叉配對與整體分母已重算。凍結版 `1a80f08`。
