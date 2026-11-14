# chameleon-defense：對「學會躲監測器的模型」做監測側防禦（第 1 週停損實驗）

## 目的與對應回報
- 對應 `reports/2026-11-14-49-relaxed-neurips-scan.md` 候選 3（Neural Chameleons 防禦）與 `2026-11-14-56-killtest-criteria-review.md` 第二梯建議（只跑第 1 週）。
- 回答兩件事：(1) 沿「逃避子空間」擴增訓練的 8 層探測器，在 1% FPR 下對留出安全概念的 TPR 是否比普通 8 層 ensemble 高 ≥20pp；(2) 不靠觸發詞的偵測器對留出概念的 AUROC 是否 ≥0.8。判準、閘門、三態見 `PREREG.md`。
- 範圍：只做監測與偵測。不訓練逃避、不跑自適應攻擊者（第 2 週）。

## 資產與授權
| 資產 | 位置 | 授權／狀態 |
|---|---|---|
| 作者程式 | `/work/ChihChungHsu/cactus8603/icml/ext/neural-chameleons`（commit 261e6547，已 clone 並讀過） | MIT；THIRD_PARTY.md 註明 obf_reps 與 RepE 資料的上游授權未解決，僅內部研究用 |
| minimal repro | `.../ext/minimal_neural_chameleons`（commit f077304）：13 個作者探測器（第 12 層，hidden 3584）、合成資料、RepE 事實資料 | 同上 |
| Checkpoint | HF `serteal/neural-chameleon-gemma_2_9b-layer_12`（公開、無授權閘；約 18.5 GB） | **尚未下載成功**：下載遇 /work 額度滿（Disk quota exceeded），已清掉殘檔；只留 tokenizer／config 小檔 |
| Gemma-2-2b checkpoint | **不存在**（作者只公開 9b；自訓需 gated 的 google/gemma-2-2b-it，未登入回 401，未嘗試登入） | 因此第 1 週用 9b |
| 第三方 HF 上的其他 chameleon 模型（ceselder/*、eekay/*） | 未使用（非作者、未審） | — |
| 環境 | 重用 `envs/inoc`（torch 2.8、transformers 4.57.6）。原本 `envs/neon` 在我工作中被他人刪除；該環境也沒有 sklearn，故邏輯迴歸改用 torch LBFGS | — |

## 資料夾
- `PREREG.md` 凍結的預註冊；`configs/config.json` 全部設定與閾值。
- `scripts/`：`common.py`（指標、bootstrap、探測器）、`build_datasets.py`、`extract_acts.py`（GPU 抽取）、`monitors.py`（ENS／ROB／INV／RND）、`selftest.py`（閘門 1 合成）、`analyze.py`（原始數字）、`aggregate.py`（機械判讀）。
- `slurm/`：`env.sh`、`01_bench.sbatch`、`02_extract.sbatch`、`03_analyze.sbatch`。`results/`、`logs/`：輸出。
- 資料、快取放 `/work/ChihChungHsu/cactus8603/icml/runs/chameleon-defense/`（repo 外，含有害合成文字，不進 git）。

## 預估
- GPU：吞吐實測 ~0.2 h、抽取 ~0.5 h（3.6 萬列）、分析 ~0.5 h；合計約 1.2 GPU 小時，上限 4 h。**皆為推估，未實測。**
- 空間：模型 18.5 GB（HF_HOME）＋快取約 2.1 GB＋資料 9 MB；遠低於 60 GB，但 **/work 群組額度目前僅剩約 19 GB，模型本身就接近用完**。

## 送出前（尚未執行，待額度恢復）
```bash
export TMPDIR=/work/ChihChungHsu/cactus8603/icml/tmp PIP_CACHE_DIR=$TMPDIR/pip-cache HF_HOME=/work/ChihChungHsu/cactus8603/icml/hf_home
# 1) 下載 checkpoint（約 18.5 GB；確認額度後再做；失敗就停）
/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('serteal/neural-chameleon-gemma_2_9b-layer_12'))"
# 2) 資料（CPU，秒級；bench 腳本內也會跑）
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/chameleon-defense
/work/ChihChungHsu/cactus8603/icml/envs/inoc/bin/python -I scripts/build_datasets.py
```

## 送出指令（寫出來，不要在準備階段執行）
```bash
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/chameleon-defense
J1=$(sbatch --parsable slurm/01_bench.sbatch)               # 10 分鐘吞吐實測；看 logs 的 rows_per_s 與 est_total_gpu_h
# 吞吐無異常再送（超過推估 2 倍就先重排）：
J2=$(sbatch --parsable --dependency=afterok:$J1 slurm/02_extract.sbatch)
J3=$(sbatch --parsable --dependency=afterok:$J2 slurm/03_analyze.sbatch)
# 之後：results/RESULT.md 為機械輸出；Opus 對照 PREREG 審。
```
（若 `--parsable` 帶 dependency 的規則不合，改為手動依序送。）

## CPU 極小 smoke test（已做）
- `scripts/selftest.py`：合成資料、秒級，閘門 1 合成部分結果在 `results/gate1_synth.json`。
- `extract_acts.py --tiny` ＋ `analyze.py` ＋ `aggregate.py`：隨機初始化的小 Gemma2（CPU，不載入真實權重），只驗證程式路徑，輸出在 run_dir 的 smoke_*（不是科學結果）。

## 風險與未確認事項
1. **checkpoint 與 8 層基線不對應**：只有 layer-12 organism；8 層 ensemble 是自訓、層集合自訂。若其他層本來就沒被擾動，ENS 會很強，Δ 難達 20pp（FAIL 也許只反映此設定）。作者 8 層 organism 未公開。
2. **重現的對照值取自 repo 已提交 CSV，不是論文表格**；HF checkpoint 是否為同一次 run 未確認。
3. 主指標只有 harm 一個留出安全概念（deception 僅 ~62 對、1% FPR 解析度不足）；結論力有限。
4. benign 概念資料是 organism 的訓練批次，E 概念位移可能比在未見資料上大；harm 才是真正未見。
5. 改寫觸發（T1–T3）下 organism 是否仍逃避完全未知；若沒有，偵測器的「留出模板」評估會退回 T0 或無法評估。
6. 偵測器本屬第 2 週，這裡因快取現成而一併算；決定權在審查者。
7. 探測器型式與作者不同（平均池化邏輯迴歸 vs 逐 token sigmoid 平均）。
8. 吞吐、GPU 小時、分析耗時都是推估；`aggregate.py` 的閘門閾值是我依原作數字與樣本數定的，需審查。
9. /work 額度近滿；下載 18.5 GB 前務必確認。
