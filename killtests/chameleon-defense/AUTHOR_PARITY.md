# 作者基準乾淨條件核對

這項診斷比較作者公開 demo 與已凍結的乾淨條件快取。正式重現閘門、觸發條件與三項停損判準沿用 [PREREG](PREREG.md)。這次診斷只量作者與本地計分管線的可追溯差異。

## 固定設計

- 使用固定的 Gemma-2-9B snapshot `c4aeb57502a8bed6a62dac57819eec27fd5e3c99`、公開 minimal demo 提交 `f077304f6bebd31ded4cc868f2115a8c81067960` 及 `cache-d42f49d` 的抽取收據。核對作者程式、資料及 probe 檔雜湊、快取資料與 probe 雜湊、模型 shard 齊備、快取對應檔與所選列索引。
- 作者 demo 使用公開合成檔的前 50 個正例與前 50 個負例。本地比較使用既有凍結 `harm_test` 的 1,000 正例與 1,000 負例。先計算兩個母體在各類別的精確文字配對重疊、原始 prompt 模板標記與本地截斷數量。各母體各類沿原順序均勻選 4 筆做 CPU 前向。
- 用作者程式的單列、未截斷、response-only 取法取得第 12 層（`hidden_states[13]`）。使用 BF16 的作者權重計分，並在同一份激活上改用本地 FP32 計分。另在本地資料比較加入既有長度限制與顯式 position ID 後的 CPU 計分，以及固定快取的 GPU 計分差。最後一項包含裝置、批次與 padding 差異。
- 背骨 Gemma 模型提供相同的 hidden states，省去詞彙 logits。從已核對的作者檔抽取原始前向函式與層索引常數，可在共用環境執行，不需 demo 的 CSV 載入模組。輸出每組均值及配對絕對誤差，不輸出文字、token ID、逐列分數、模型生成或正式閘門 PASS／FAIL。這個小樣本不提供穩定 TPR 或完整論文資料重現。

## 執行

協調者提供凍結提交，並確認磁碟空間、工作樹與佇列。預檢只核對雜湊、資料計數及路徑。正式診斷使用 CPU 分區，申請 16 CPU、64 GiB RAM、兩小時，約 24 次單列前向，JSON 輸出遠低於 1 MiB。時間仍需由本次作業量測。

```sh
bash killtests/chameleon-defense/slurm/06_author_parity_cpu.sbatch FULL_COMMIT_SHA --check-only
sbatch --parsable killtests/chameleon-defense/slurm/06_author_parity_cpu.sbatch FULL_COMMIT_SHA
```

結果為 `results/author_parity/<job-id>.json`。作業只讀先前模型與快取，使用凍結程式副本。可用合成測試驗證 token 池化、背骨等價性及快取列對齊。
