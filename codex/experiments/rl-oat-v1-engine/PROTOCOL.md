# Oat-Zero v1 公開引擎診斷

arXiv 2025-03-26 的 Table 4 已列 Oat-Zero-1.5B 的 MATH500 74.2% 和 Minerva 25.7%。同日的公開安裝說明 `08db9b8` 指定 Python 3.10、vLLM 0.7.2 與 Oat 0.0.9。同一版本的評估腳本使用 vLLM，Oat 的發行 metadata 指定 Transformers 4.48.2。評估只用 vLLM 和已固定的評分 subprocess。原論文的實際執行清單仍未公開。 [Table 4](https://arxiv.org/html/2503.20783v1#A1.T4) · [同日作者安裝說明](https://github.com/sail-sg/understand-r1-zero/blob/08db9b810ace3ed128bbee85e780b418d72a156e/README.md#L99-L107) · [公開評估命令](https://github.com/sail-sg/understand-r1-zero/blob/08db9b810ace3ed128bbee85e780b418d72a156e/README.md#L170-L176) · [Oat 0.0.9 發行 metadata](https://pypi.org/pypi/oat-llm/0.0.9/json)

作者 `08db9b8` 和現用 `dfca49d` 的 evaluator、評分器、MATH500／Minerva Arrow 與各自 metadata 均是相同 Git blobs。Evaluator 為 `0aa3c46129dad3dde5fe77a4d417eed18db96aed`，評分器為 `2ec16e8bda938fdc11d98af8ed86a072a520e39d`，MATH500 Arrow 為 `9836ed58862ff87846530acd0f543933d62239a4`，Minerva Arrow 為 `0a666bb23ca63f855b415b5c49d645d10f3949c6`。工作從固定 `0ea7e89` 封存先前驗證的 wrapper，生成及評分程式維持同一路徑。 [作者 v1 原始碼](https://github.com/sail-sg/understand-r1-zero/tree/08db9b810ace3ed128bbee85e780b418d72a156e) · [公開 Oat 配對 A 流程](../rl-pair-a-author/PROTOCOL.md)

## 固定比較

- 一次各跑完整 MATH500 500 題、Minerva 272 題，照固定素材自身順序。保留原 Qwen-Math 模板、BF16、貪婪取樣、3,000 新 token、4,096 context 及原生 EOS 151643。記錄作者 evaluator 的 `time_ns` seed。兩面板沿用 26621 的範圍。
- 推論換到 0.7.2 預設 V0 與該公開配方的 Transformers 4.48.2。使用發行套件要求的 Torch 2.5.1／CUDA 12.4，以及正式執行前登記的 package inventory SHA-256。這個比較涵蓋推論環境，無法個別歸因於 vLLM、Torch 或 tokenizer 版本。
- 評分保持 26621 的原 wrapper 來源和固定套件，包括 Math-Verify 0.9.0。作者 `pyproject.toml` 只指定 Math-Verify 的 ANTLR extra，兩版來源均缺確切 Math-Verify 版本。歷史評分版本敏感度另行執行，需事先凍結同批文字及比較的套件組。 [作者依賴聲明](https://github.com/sail-sg/understand-r1-zero/blob/08db9b810ace3ed128bbee85e780b418d72a156e/pyproject.toml#L27-L31) · [Math-Verify 發行紀錄](https://pypi.org/project/math-verify/0.9.0/#history)
- 只輸出兩面板的正確數、母數、準確率、與論文百分點差、原 ±2 百分點範圍指標、完成原因分布及來源憑證。這個診斷維持 26621 的判定和短預算先決條件。

## 離線執行

使用隔離的 shared `envs/oat-vllm072`，官方 Linux x86_64 vLLM 0.7.2 wheel SHA-256 為 `51fcb7d60149f75ec85d457fab6ccd88a2b1a6344c28aad46b0f8776b29e9d47`。Torch 2.5.1 Linux CPython 3.10 wheel SHA-256 為 `71328e1bbe39d213b8721678f9dcac30dfc452a46d586f1d514a6aa0a99d4744`。安裝時先核 wheel，再保存 `uv pip freeze` 為環境內 `installed-packages.txt`。安排一張 H200、48 GiB 主記憶體、8 CPU、30 分鐘上限。現有資料與權重只讀，JSON 寫入共享 `icml/runs/rsd-pair-a-v1-engine/`，本次成果預期小於 1 MiB，不存逐題資料。 [vLLM 0.7.2 PyPI](https://pypi.org/project/vllm/0.7.2/#files) · [Torch 2.5.1 PyPI](https://pypi.org/project/torch/2.5.1/#files)

凍結後依序執行資產、推論環境 CPU 預檢，接兩筆合成題 GPU smoke。兩面板作業等待 smoke 成功並核同凍結、同 inventory 的 receipt。

[codex] 2026-11-15 12:36（台北）。獨立程式審查核對相同評估封存、兩面板順序、生成配置與隔離評分。三個單元測試及啟動腳本語法通過。

```bash
stage=codex/experiments/rl-oat-v1-engine
bash "$stage/run.sbatch" "$freeze" --check-assets
bash "$stage/run.sbatch" "$freeze" --check-runtime "$inventory_sha"
sbatch --parsable "$stage/run.sbatch" "$freeze" smoke "$inventory_sha"
sbatch --dependency="afterok:$smoke_job" "$stage/run.sbatch" "$freeze" run "$inventory_sha" "$smoke_job"
```

## 讀取紀錄

- [codex] 2026-11-15 12:52（台北，12:52:08）。已讀 `icml/runs/rsd-pair-a-v1-engine/26652.json`，凍結 `1432b5d`。 [研究進度報告](../../reports/2026-11-15-topic-status-codex.md#rl-稀疏決策蒸餾)
- [codex] 2026-11-15 12:52（台北，12:52:16）。Astra 獨立讀取同一份凍結彙總。
- [codex] 2026-11-15 12:52（台北）。重讀既有 `aggregate_gate0_A.json` 的歷史短預算資格摘要。
