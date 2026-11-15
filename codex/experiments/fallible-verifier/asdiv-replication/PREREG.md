# ASDiv-A 外部金標面板

用公開小學算術做外部驗證，接續強驗證器的共同誤收、條件訊號與等風險省費研究。ASDiv-A 的題目為一至兩次運算，資料分布有別於 GSM1k。本協定包含固定資料、候選生成條件與驗證規格。 [ASDiv 作者論文](https://aclanthology.org/2020.acl-main.92/) · [作者固定版](https://github.com/chaochun/nlu-asdiv-dataset/tree/55790e5270bb91ccfa5053194b25732534696b50/dataset)

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

## 候選與判讀

- 接受新作答配置的前提是以凍結的 GSM1k 暖機驗證資料通過技術資格。正式工作驗證該作業已成功、憑證雜湊與來源 log 一致，且重新核對 850/1,000 嚴格可解析及每驗證器 980/1,000 可送入雙映射的必要條件。此檢查仍與 ASDiv 題目正誤隔離。 [指令模型暖機](../instruct-candidate-coverage/PREREG.md)
- ASDiv 每題由同一凍結 Qwen2.5-7B-Instruct 與原 Oat-Zero 各抽兩筆。沿用各自的暖機提示、每題兩個配對種子、temperature 0.6、top-p 0.95、BF16、768 tokens 及各自的原生停止規格。作答器只收公開題文與清冊。組裝階段依原最終數字框、正常完整停止及作者精確數值判定正誤。不可解析或截斷候選留未知，保留原面板母數。
- 弱／強端仍採原 Qwen3-0.6B、Qwen3-32B-AWQ、2,048 tokens 界線、中性雙映射與原棄權界。沿用原 250 題重新擬合、從 1,000 暖機候選兩端同額攤銷 0.135、強判決／金標 1:10 的規格。只在 747 題鎖定留出判讀共同誤收、條件 Brier、原六項風險與 20% 省費門檻，及完整暖機資訊參考。保留 85% 金標、至少兩真類各 50 的原判讀守門。資料題群與作答器皆更換，結果是 ASDiv-A 下重新暖機的方法外部驗證。 [原留出判讀](../holdout-evaluation/EVALUATION_INTERFACE.md)

一張 H200 依序跑兩個作答器與兩種驗證器。估 **15–35 GPU 分鐘**，上限 2 小時，保留產物估 **<40 MiB**，只留共享磁碟。兩種模型快取既有，CPU 可做資料及彙總。完成並登記讀取後交獨立複核。
