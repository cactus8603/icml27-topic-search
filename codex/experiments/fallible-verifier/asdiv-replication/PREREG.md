# ASDiv-A 外部金標面板

公開小學算術的外部驗證資料，用來接續強驗證器的共同誤收、條件訊號與等風險省費研究。ASDiv-A 的題目多為一至兩次運算，資料分布有別於 GSM1k。候選生成與驗證規格另行凍結，目前只準備資料。 [ASDiv 作者論文](https://aclanthology.org/2020.acl-main.92/) · [作者固定版](https://github.com/chaochun/nlu-asdiv-dataset/tree/55790e5270bb91ccfa5053194b25732534696b50/dataset)

- 固定 `55790e5270bb91ccfa5053194b25732534696b50` 的 XML 與五份作者 ASDiv-A 算術索引。納入可抽出一個確切數值、答案與公式右側一致、四則公式以有理數重算也一致的一運算或兩運算題。單位保留在原始資料溯源，數值比對用 Decimal。資料選擇先於本題的作答與驗證輸出。
- 與完整 GSM1k、GSM8K train/test、先前兩輪比較 NFKC／casefold／去空白的精確題文及連續五字元 shingles 的 Jaccard ≥0.80，另核對 ASDiv-A 內部近重複。固定選擇 seed `202611151601`，先固定每個運算層的題目順序與人數，再用前段作暖機，後段留出。暖機一／兩運算 **180／70**，留出 **539／208**。保留全體通過的兩運算題，藉此維持新面板的兩個難度層。適用範圍限固定比例的 ASDiv-A，預訓練資料重疊未量。
- `questions.jsonl` 的表只有題文、題目雜湊、暖機／留出標記。`references.jsonl` 存相同題目 ID 的數值答案。兩檔指紋分別寫入 `prepared.json`。本次準備清冊 SHA-256 `110881d5e9a7ba12d2776a30dbebe8ef919f76a81ef39b2a829790ffd2b2b8d8`。

```bash
shared=/work/ChihChungHsu/cactus8603/icml
stage=codex/experiments/fallible-verifier/asdiv-replication
source="$shared/data/asdiv-official/55790e5270bb91ccfa5053194b25732534696b50"
"$shared/envs/vllm033/bin/python" -IB "$stage/test_prepare.py"
"$shared/envs/vllm033/bin/python" -IB "$stage/prepare.py" --source-dir "$source"
```
