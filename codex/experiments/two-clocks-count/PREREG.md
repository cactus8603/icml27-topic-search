# 同母體配對的唯一本數對照

三臂探索 `26523` 把早期網格補密後，每組都可定位內點。原對照以不同類別分布、條件策略與批次比較訓練臂。本實驗在另一批合成樹上只改唯一訓練樣本數，觀察有限資料的時鐘是否提早。此為前瞻的機制診斷，沿用原試驗的正式現象、放大與上限紀錄。 [三臂診斷](../two-clocks-cohort/PREREG.md) · [原判準](../../../killtests/biased-gen-cfg/PREREG.md)

## 固定樣本與訓練

- 從原凍結階層資料取出字面轉移文法 $M$。對每個固定索引，用作者生成函式從類別 0 的同一樹分布抽一筆。依固定規則排除原 40,000 筆中相同的葉序列，及新集合重複葉序列。收齊 35,000 筆後，以獨立固定亂數打亂再分成七個互斥的 5,000 筆區塊。資料是該分布抽樣後的唯一葉序列面板。
- 訓練區塊為 0–5。各區塊的前 833 筆與同一區塊的 5,000 筆形成嵌套配對。每個區塊的前 700 筆是兩臂共用的自身訓練輸入。第 6 區塊提供一致的共同未見輸入。
- 兩臂同用凍結階層 transformer、null 條件、同一初始化 seed、Adam、噪聲 RNG、學習率排程和 85 筆固定 batch。每個虛擬 epoch 固定 10 次更新，各用相同均勻亂數流在其支撐集合有放回抽樣。長程排程仍是 20,000 個虛擬 epochs，實跑至預先固定的第 192 虛擬 epoch。新取樣和固定更新量有別於早期實驗的遍歷一輪定義。沒有任何模型結果參與資料構成。

## 判讀

- 共用原八個擴散時間、遮罩、epsilon cosine、每個擴散時間正規化及三點平滑。使用新早期網格 `1,2,4,8,12,18,24,30,36,44,54,66,82,101,125,155,192`。只有兩臂各有內部最低點並符合原 10% 回升的 pick pair 進比較，至少 10 個共同有效配對。此處只使用 pair 曲線，不使用評估器內的均勻根先驗 BP 欄位。
- 主量為共同有效對的平均 log 最低點時間差 $L_{5000}-L_{833}$。以同批六 pick 重抽在兩臂共享權重，2,000 次，95% 區間。要求至少 95% 重抽有可評估配對。點估計至少 0.4463 且下界大於零時支持原量級的樣本數分離。區間上界小於零視為此設定方向相反。上界至少零、低於 0.4463 時低於原參考幅度。報告前置失敗、邊界情況和各臂幾何時間。
- 解讀限同一固定文法、類別、樣本與抽樣方式。若要證成實際條件分支的機制，還要固定樣本數來比較條件策略。正式精確分數差距 G3 沿用 `26383` 的判讀。

## 資源

新資料不到 1 MiB，一張 H200，估 45–70 分鐘，上限 150 分鐘。最多 12 條訓練軌跡 × 17 個 fp16 checkpoint，暫存預計約 6.5 GiB。作業只長期保留小型彙總，結束時清理本作業的暫存。24 GB 卡仍需各自核對吞吐和主機記憶體。

```bash
stage=codex/experiments/two-clocks-count
shared=/work/ChihChungHsu/cactus8603/icml
export TMPDIR="$shared/tmp" BG_EXT="$shared/ext/biased-generalization" OMP_NUM_THREADS=1
export COUNT_CLOCK_TEST_DATA="$shared/data/biasgen-hier/hier_q6_l4_s1.0_qeff4_seed0_n40000.npy"
"$shared/envs/biasgen/bin/python" -IB "$stage/test_count_clock.py"
bash "$stage/run.sbatch" FULL_COMMIT --check-only
sbatch --parsable "$stage/run.sbatch" FULL_COMMIT
```
