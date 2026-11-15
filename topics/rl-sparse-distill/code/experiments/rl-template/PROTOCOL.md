# RL 配對 B 的模板敏感度

[codex] 2026-11-15 台北。MATH500 的公開規格重現未達 Gate 0(a)。正式重現仍依原判讀，這是獨立樣本的模板診斷。

- 從同一作者快取原 MATH 公開測試集的 5,000 筆出發。先以 Unicode NFKC、大小寫與空白正規化，排除配對 B 的三個重現題集及本 repo 的 1,453 個短預算題目，再對相同題文去重。用固定 seed 2026111507 的雜湊排序選前 768 題。選擇只依題文，固定來源 SHA-256。詞面排除不能識別所有改寫，也無法判斷模型訓練是否看過這些公開題。
- 比較 raw `abel` 與作者 `--apply_chat_template` 路徑。兩臂共用同一 RL checkpoint、vLLM 0.33.0、BF16、temperature 1、top-p 0.95、16,000 token、原停止條件。每題在兩臂用相同的固定取樣 seed。寫入同一引擎請求，記錄每臂題數、截斷及判分逾時。
- 使用凍結的作者答案抽取、兩次標答正規化與等價判分。分母保留同一批題，逾時記錯。主估計為 chat 減 raw 的正確率百分點差，搭配 6,000 次整題配對重抽的雙尾 95% 區間。這個估計只量模板對新題表現的敏感度，無法確定作者未公開的表列指令或恢復短預算資格。

申請一張 H200，最長 30 分鐘。前一輪的 1,447 題 RL 推論約 4 分鐘，這次兩臂合計 1,536 次生成，第一輪硬體實測僅供估算。暫存預算低於 1 GiB。原始生成及題目暫存於單次作業目錄，結束後清理，只留數字彙總 `icml/runs/rsd-template/<job-id>.json`。凍結後由非實作者先行複核。

```sh
bash codex/experiments/rl-template/run.sbatch FULL_COMMIT_SHA --check-only
sbatch --parsable codex/experiments/rl-template/run.sbatch FULL_COMMIT_SHA
```

- [codex] 已讀 `icml/runs/rsd-template/26515.json`，2026-11-15 07:23。凍結版 `743bda8`，Astra 獨立複核與整合核對。
