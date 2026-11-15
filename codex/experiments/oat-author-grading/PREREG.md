# Pair A 同一輸出的作者計分診斷

用歷史 Pair A 的 RL／Base greedy 作答，比較作者計分、歷史的停止規則，以及作者答案欄帶來的差異。這次重用已產生的輸出，原正式測試、結果及門檻留在原實驗。歷史來源為 RL 26290 與 Base 26291，各題一次、temperature 0、最多新生成 3,000 token、4,096 context，沿用當時 BF16、Qwen-Math 模板、vLLM 和模型版本。作者評估腳本使用相同模板與上限，兩次執行在 vLLM 版本、seed 設置及停止 token 配置可能有差異。這次沒有重新生成，作者公開程式是計分規格的來源。 [歷史配對](../../../killtests/rl-sparse-distill/PREREG.md) · [作者評估程式](https://github.com/sail-sg/understand-r1-zero/blob/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87/evaluate_model.py)

來源使用固定發布版本的 MATH500、Minerva、OlympiadBench Arrow。以三個來源各自唯一的逐字題文連接歷史 ID 與作者 Arrow 列序號，再用作者函式和歷史模板驗證完整提示相等。僅在實際交集做重評。分別報同一題文的交集與歷史答案字串逐字出現在作者答案欄的子集。若作者答案欄是陣列，子集仍允許其他作者接受的替代答案。NFC 與空白正規化的比對只產出資料來源統計。重評不使用正規化後的字串替代原答案。 [固定版發布資料](https://github.com/sail-sg/understand-r1-zero/tree/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87/datasets/evaluation_suite)

對同一輸出用歷史共用答案先重算四個條件。兩個計分器是既有 Math-Verify 規則與作者未修改的 `boxed_reward_fn(fast=False)`。兩個停止規則是僅計正常停止及允許所有現有輸出。第五欄改為作者計分器、允許所有輸出並使用原發布的完整答案欄。以既有 verified receipt 驗證歷史分片、模型及計分。重算的歷史方法須逐列與 verified 判分一致。輸出分來源的各條件對數及配對改判增加和減少數。保留未正常停止且達生成上限、未正常停止且低於上限的個數。歷史分片僅存是否任務正常停止，無法由低於上限的記錄重建具體結束原因。

作者發行沒有固定當年的所有解析依賴。使用現有 vLLM 環境的 Python、PyArrow 和 SymPy、歷史配對的 Math-Verify 0.9.0 和對應 ANTLR，再加作者要求的 pylatexenc 2.10。此環境能檢查相同完成文本對發布版計分規則的敏感度，不能以此保證原作者執行環境、原生成或未匹配題目的結果。作者 repo 有 MIT 根授權，評估與評分檔保留 Apache-2.0 檔頭。原始公開資產與套件放在共用 `icml/tmp`，repo 留版本與雜湊，暫存小於 2 MiB，沒有下載權重。 [作者根授權](https://github.com/sail-sg/understand-r1-zero/blob/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87/LICENSE.txt) · [評分程式檔頭](https://github.com/sail-sg/understand-r1-zero/blob/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87/understand_r1_zero/math_grader.py) · [pylatexenc 2.10](https://pypi.org/project/pylatexenc/2.10/)

## 離線預檢與凍結後重評

```bash
shared=/work/ChihChungHsu/cactus8603/icml
topic=codex/experiments/oat-author-grading
stage="$shared/tmp/oat-author-grading-dfca49d"
python="$shared/envs/vllm033/bin/python"
export TMPDIR="$shared/tmp" HF_HOME="$shared/hf_home"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$python" -B "$topic/prepare.py" "$stage"
"$python" -B "$topic/regrade.py" --assets "$stage" --check-only
```

預檢對歷史分片和判分檔只核對檔案 SHA-256，另核對中繼資料、收據、公開來源與合成評分測試。資產備妥後全程離線。以下命令須在程式、設定及本文件複核並凍結後，才可開啟歷史輸出與逐題判分。作業使用 `h200q` 的八個 CPU worker、12 GiB、30 分鐘，GPU 配額為零。只在 `results/<job-id>/author-grade-aggregate.json` 寫一個彙總收據，含工作編號、凍結雜湊、資料交集對應雜湊、環境與台北時間讀取紀錄。

```bash
freeze=$(git rev-parse --verify HEAD)
sbatch "$topic/run.sbatch" "$freeze"
```

## 讀取紀錄

* [codex] 已讀 `results/26525/author-grade-aggregate.json`，2026-11-15 09:04（台北）。凍結版 `ed42f9d94d1de04d03236b5834eabd4325eece36`。
* [codex] 獨立複核已讀同一彙總，2026-11-15 09:05（台北）。
