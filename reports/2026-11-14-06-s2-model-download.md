## 回報：下載結果

**Workspace**：`/work/ChihChungHsu/cactus8603`（`KVBENCH_WORKSPACE` 未設，用 repo 預設，即 repo 的上層目錄）。這也是我的 `$HOME`。repo 的約定是 `models/`、`data/`、`src/` 放在 repo 上層。

**下載方式**：用 repo 既有腳本，執行環境是 `icml/envs/s2/bin/python`（Python 3.11、huggingface_hub 0.36.2）。本機沒有 python3.11、`hf` 或 `huggingface-cli`。

**驗證**：
- 無 `.incomplete` 或 `.part` 檔。
- `model-lock.json` 四個模型皆為 `downloaded_and_hashed`。
- 三個 1B 權重的 sha256 與 `setup-env-3090.sh` 內的值相符。
- repo 內沒有被修改的檔案，也沒有 commit。

| 名稱/ID | 用途（引用處） | 大小 | 存放路徑 | 狀態 |
|---|---|---|---|---|
| QwQZh/gated_attention @aad415c（baseline、gate_headwise、gate_elementwise 三個 1B，30 個檔案） | `setup-workspace.sh`、`setup-env-3090.sh`、`configs/baseline-freeze.json`（legacy_models 三個） | 9.7G | `models/gated_attention/` | 本次下載完成 |
| Qwen/Qwen3-8B @b968826 | `benchmark-models.json`、`run_official_baselines.py` | 16G | `models/baseline-suite/Qwen3-8B` | 本次下載完成 |
| nvidia/KVzap-mlp-Qwen3-8B @bd5c591 | 同上 | 約 0.35G | `models/baseline-suite/KVzap-mlp-Qwen3-8B` | 本次下載完成 |
| Jang-Hyun/Fast-KVzip @67ce126（只取 qwen3-8b/q4_dim16_sink16.pt） | 同上 | 約 0.19G | `models/baseline-suite/Fast-KVzip` | 本次下載完成 |
| marcusguhao/IndexMem-Qwen3-8B @c92e04c | 同上 | 約 0.08G | `models/baseline-suite/IndexMem-Qwen3-8B` | 本次下載完成 |
| 五個 GitHub source snapshot：NVIDIA/kvpress、NVlabs/RocketKV、Janghyun1230/FastKVzip、Chooovy/IndexMem、THUDM/LongBench（皆為固定 commit，`benchmark-sources.json`） | `fetch-benchmark-sources.py`；`setup-workspace.sh` | 37M | `src/benchmark-sources/`（含 source-lock.json） | 本次下載完成（腳本已逐檔雜湊） |
| PG-19（deepmind/pg19 @4d28bd7 清單，來源是 GCS bucket）：test 100 本加 validation 50 本 | `collect-pg19.py`、`download-pg19.sh`、`prepare-development-data.py` | 57M | `data/pg19/`（含 manifest.json） | 本次下載完成（test 與 validation 子目錄共 153 項，含清單檔與 manifest） |
| LongBench v1（zai-org/LongBench @5e628be 的 data.zip，加 THUDM/LongBench @2e00731 的設定）：16 個英語／程式子集 | `collect-longbench.py`、`benchmark-suite.json`（longbench-native4k-dev） | 272M | `data/longbench-v1/` | 本次下載完成 |
| SCBench（microsoft/SCBench @283310b）：5 個子集 | `collect-scbench.py`、`prepare-development-data.py`（`datasets` 元組列了 scbench，且硬性要求 manifest 存在） | 242M | `data/scbench/` | 本次下載完成 |
| RULER 資料（NVIDIA/RULER @db43f92 的生成輸入，ruler-dev4 與 ruler-dev13） | `prepare-baseline-ruler.py`、`collect-ruler.py`、`benchmark-suite.json` | 未估 | 預定 `data/ruler*` | **跳過**，原因見下 |

**總下載量**：約 26.8 GB（Qwen3-8B 16G、gated_attention 9.7G、其餘合計約 1.1G）。

**磁碟空間**：`/work` 剩 20T（共 28T）。本帳號用量從 146G 增到 172G，配額 1536G，仍剩約 1.36T。`/tmp` 只剩 4.3G，我沒有往那裡寫大檔。

**需要人決定的事項**
1. **RULER 沒有下載。** 它是用 tokenizer（預設 `models/gated_attention/1B_baseline`）在本機生成的資料，不是下載資料。`prepare-baseline-ruler.py` 需要 `src/benchmark_common` 和生成器的相依套件，還要抓 Paul Graham essays、SQuAD、HotpotQA。README 的 H200 流程是改用 `ruler13-dev208.tar.gz` 與 `portable-assets.json`。這兩個檔案在原工作機 `/ssd4/cactus8603/icml/runs/transfer/h200-dev-assets/`，本機沒有 `/ssd4`。需要你提供這個 tar，或決定在本機生成。
2. **模型需求在 repo 內不一致。** `setup-workspace.sh --download-models` 只下載 headwise 加 baseline 的 tokenizer，但 `baseline-freeze.json` 要求 baseline、headwise、elementwise 三個模型。我下載了整個 repo 在該 revision 的內容（superset）。若只需要 headwise，elementwise 與 baseline 權重（約 6.9G）可不要。
3. **Qwen3-8B 重複。** `icml/models/Qwen3-8B`（16G）是另一個專案的副本，revision 未知，所以我沒有沿用，而是依 repo 的固定 revision 另下載。若確認是同一版，可以刪掉其中一份（我沒有動它）。
4. **環境沒有建。** `setup-workspace.sh` 要 python3.11 加 pinned requirements（`envs/kvbench-author`、`kvbench-qwen`）。這台沒有 python3.11，我也沒跑這支腳本。腳本若重跑，會驗證並沿用現有的 sources 與 models。
5. **collect 腳本的預設路徑有問題。** `collect-longbench.py` 預設輸出到 `/ssd4/...`，`collect-pg19.py` 與 `collect-scbench.py` 的預設路徑也不是 `$WORKSPACE/data`。我都用 `--output`／`--root` 明確指到 `data/`。之後若有人不帶參數重跑會寫到別處。
6. **沒有 sibling 專案 PG-19 的比對。** `icml/data/pg19` 是另一個專案的 test 100 本。我沒有把它與新下載的版本做 hash 比對。

下載用的暫存腳本與 log 在 `/tmp/claude-1047/-work-ChihChungHsu-cactus8603-icml27-s2-gated-sink-kv/bd65f9a7-961a-4b8d-adce-42df28fbb74b/scratchpad/dl.sh` 與同目錄的 `dl.log`。
