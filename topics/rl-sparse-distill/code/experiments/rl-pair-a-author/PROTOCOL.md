# Oat-Zero 配對 A 的公開規格重現

這次只重現原 Gate 0(a) 的 RL greedy 數字。以 Oat-Zero 論文 Table 4 的 MATH500 74.2% 與 Minerva 25.7% 為參照，兩者皆須在 ±2 個百分點內。`26525` 對歷史生成的重評照原報告留存。正式的短預算截斷及 crossover 另循原協定。 [原計畫](../../killtest/PREREG.md) · [論文 Table 4](https://arxiv.org/html/2503.20783v2#S2.T4) · [歷史重評](../oat-author-grading/PREREG.md)

## 固定路徑

- 沿用 `dfca49dd460ee7cc8e4a5a162c876a7fd6993b87` 的公開 1.5B 評估入口及本地 RL snapshot `a98e477854071157a450e57dd45fd0684b6fa38a`。直接執行雜湊固定的作者 `main` 和 `qwen_math` 模板。從原始 Arrow 依列讀取 `problem` 和完整 `answer`，保留作者資料集索引中的 `math`、`minerva` 順序。各 500 與 272 筆，每題一次，無抽樣或排序替換。 [作者命令](https://github.com/sail-sg/understand-r1-zero/blob/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87/README.md#evaluation) · [資料集順序](https://raw.githubusercontent.com/sail-sg/understand-r1-zero/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87/datasets/evaluation_suite/dataset_dict.json)
- Python 3.10.20、vLLM 0.8.4、Torch 2.6.0/CUDA 12.4、Transformers 4.51.3、NumPy 1.26.4。原入口 BF16、greedy、top-p 1、logprobs 2、prefix cache、3000 新 token、4096 context、32 GiB vLLM swap space。保留從 `time_ns()` 取得的原 seed 並記錄。使用 checkpoint 原生 EOS 151643，核對沒有額外終止條件。歷史 wrapper 加入了 `<|im_end|>` 151645，這次依發布入口重新生成。完整歷史機台及套件鎖未發布。 [發布環境](https://github.com/sail-sg/understand-r1-zero/blob/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87/README.md#install) · [vLLM 0.8.4 安裝文件](https://docs.vllm.ai/en/v0.8.4/getting_started/installation/gpu.html)
- 另以 `26525` 核對過的 Python 3.12、PyArrow 及數學依賴執行固定版本 `boxed_reward_fn(fast=False)`。生成與評分程序以匿名管道交換資料，評分與批次順序採原入口行為。缺回答、額外終止或題序不符時中止，所有完整作答均進作者計分。只寫彙總與版本、資料、顯示行為的憑證至 `icml/runs/rsd-pair-a-author/<job-id>.json`。

## 預檢與排程

已存權重約 2.88 GiB，無需重下載，正式入口重算權重內容 SHA-256。獨立 GPU 環境在 `icml/envs/oat-vllm084/`，評分沿用現有 `vllm033` CPU 路徑。預檢只驗證 Arrow 雜湊、結構與筆數，使用合成題透過原入口、真實 tokenizer、SamplingParams 和獨立作者評分程序。首次 H200 smoke 用同模型與引擎，每筆最多 16 token，只跑兩筆合成題、未開啟發布題；收據只含完成型態和環境。正式作業同版本核對收據並等 smoke 成功才啟動。各申請一張 H200、8 CPU、48 GiB RAM，smoke 上限 10 分鐘、正式 30 分鐘。772 次 greedy 估約 3–10 分鐘，GPU 峰值待實測。一次只留一份小型彙總，編譯暫存於獨立作業目錄。

```bash
stage=codex/experiments/rl-pair-a-author
bash "$stage/run.sbatch" "$freeze" --check-only
bash "$stage/run.sbatch" "$freeze" --check-runtime
smoke=$(sbatch --parsable --time=00:10:00 "$stage/run.sbatch" "$freeze" smoke)
smoke=${smoke%%;*}
sbatch --parsable --dependency="afterok:$smoke" "$stage/run.sbatch" "$freeze" run "$smoke"
```

合成 smoke 的 Slurm 日誌保留完整引擎輸出。引擎初始化未完成時，每八秒列一份 Python 堆疊。正式作業的日誌只保留階段和彙總指紋。`26607` 在初始化結束前收到 SIGKILL。同設定診斷 `26612` 顯示 CUDA V1 worker 匯入 `triton.runtime.build` 時缺 `setuptools`，引擎收到 worker 的致命訊號後終止主程序。CPU 已重現同一匯入錯誤。隔離環境補 `setuptools 80.9.0`，預檢直接測該匯入並核對環境指紋，生成與引擎參數沿用原設定。

## 讀取紀錄

- [codex] 2026-11-15 12:14（台北），Codex 和 Astra 子代理獨立讀取 `26621.json`，凍結版本 `0ea7e896d68e238a360402607a1573185430c2d2`。
