# Λ 西語 Gate 2 離線匯入

本工具供 `26567` 評審封包完成後使用。它核對 `26566` 生成版本、`26567` 評審版本、封包雜湊、600 筆題目／樣本位置、每筆原始請求及回覆。作者 `6964328` 的判分函式按原樣處理 `gpt-4.1-mini-2025-04-14` 的單 token top-logprobs。機率加權所用的 0–100 數字 token 質量不足 0.25 時，該筆無分數，整組正式憑證暫不產生。拒評、缺列、錯版、重複回覆或請求雜湊不符也會停止匯入。 [封包格式](../lambda-author-review/README.md) · [正式 Gate 2](../../killtest/phaseb/README.md)

`native.jsonl` 每列有三個欄位。`id` 沿用 `author_requests.jsonl` 的短 ID。`request_sha256` 是該列 `request` 物件依 UTF-8、排序鍵、緊湊 JSON 且附一個換行的 SHA-256。`response` 放未改寫的原生 Chat Completions response body，含精確日期版本、completion ID 和單 token 的 top-logprobs。保留原始交付來源供獨立核對。檔案指紋與格式可驗證交付一致性，供應來源仍需由審查者核對。

在 importer 凍結且收到合格來源後，以既有 `inoc` 環境執行。`packet_sha` 為 `26567` 封包 `receipt.json` 的 SHA-256，`native_sha` 為已核對交付的 SHA-256。原始封包、供應原始回覆及數值憑證留在共享實驗目錄。

```bash
stage=codex/experiments/lambda-author-import
tool="$stage/import_scores.py"
python=../icml/envs/inoc/bin/python
"$python" -IB "$tool" --freeze "$freeze" --packet "$packet" --packet-sha256 "$packet_sha" \
  native --responses "$native" --responses-sha256 "$native_sha" --evidence "$evidence"
```

標準輸出包含沿用原協定的機械預判、憑證 SHA-256、各臂均值及兩個效應的 95% 區間。區間用固定種子共同重抽 20 個題目群 10,000 次，每次保留同題三臂與原本各 10 筆。正式門檻仍用原協定的點估計及方向，最小效應為 39.0 與 39.15 個百分點。獨立審查完成後，用其核對過的憑證雜湊執行原正式入口。

```bash
PYTHONPATH=killtests/inoculation-lambda "$python" -B -m phaseb.feasibility gate2 "$evidence" \
  --binding 1a0873aa84988bf845d9e7edd2ab57c8873c4a359c2459eef99fcd66919ae1c0 \
  --evidence-sha256 "$independently_reviewed_sha"
```

另有 `pilot-agreement`。兩位評者只收同一份 `pilot.json`，各交回 `reviewer` 代號、`pilot_sha256` 和 `scores_0_100`。`scores_0_100` 按完整 60 個盲 ID 各填一個 0–100 分數或 `null` 棄權。全數有分才報 ICC(A,1)、完全相同數與平均絕對差。棄權時報兩評者各自棄權數及共同有分數量。Astra 或人工盲評皆屬診斷，這個入口只輸出一致性摘要。

```bash
"$python" -IB "$tool" --freeze "$freeze" --packet "$packet" --packet-sha256 "$packet_sha" \
  pilot-agreement "$rating_a" "$rating_b"
"$python" -B -m unittest discover -s "$stage" -v
```
