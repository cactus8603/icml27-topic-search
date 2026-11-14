# REVIEW：compositional-retrieval（E0–E2）

審查者：Opus｜日期：2026-11-14｜範圍：PREREG.md、configs/params.json、scripts/*、slurm/*、README｜依據：reports 46 §6、56 通用規則與第二梯建議。
只做了閱讀與 CPU 上的極小檢查，包括在 COCO val2017 標註上實際模擬 E2 的查詢數。沒有送出任何 job，也沒有 commit。

## 結論：**修改後可送**
- stage1（閘門 G1–G3 與 P1）要先處理必修 1–3。
- stage2（E2／P2）還要再處理必修 4。

## 這題怎麼判（給使用者）
- **閘門**（任何一項失敗 → 未能測試，給 3 個工作天修）：
  - G1：單元測試、與官方腳本逐列比對誤差 ≤ 1e-4、細節抽取覆蓋率 ≥ 95%。
  - G2：CS-CLIP 的 COR 在三個子集都落在 58.2／50.4／42.2 ±3pp。
  - G3：bootstrap 半寬 ≤ 5pp，且改 batch 重跑的誤差 ≤ 1e-5。
- **P1（主停損）**：CLIP B/32 + LLR 加總在 COCO(Qwen) 的 COR，門檻 45%。判法依必修 2 的定案版本。
- **P2**：holistic 的成對錯誤率差 Δ = err(k_top) − err(1)。下界 ≥ 5pp 且相鄰 k 的錯誤率下降 ≤ 2pp 為 PASS；上界 < 5pp 為 FAIL。
- **整體**：任一主指標 FAIL → 停損；兩者都 PASS → 繼續；其餘交使用者。
- 判讀由 `aggregate.py` 機械輸出。我核對過，門檻數字與 PREREG 一致；CI 的 2.5／97.5 百分位寫死在程式裡，與 params 的 α=0.025 吻合。

## 對照 56 號通用規則
| 項目 | 狀態 |
|---|---|
| 唯一主指標與公式 | 有 2 個主指標，各自單一、公式明確。兩者是「都要 PASS 才繼續」的 AND 關係，嚴格說不需要 Bonferroni；現行做法除以 2，可以接受 |
| 三態、灰區唯一加碼 | P1 用 CS-CLIP 救援當加碼；P2 用第二組 seed 的查詢當加碼 |
| 閘門順序（正確性 → 重現 → 雜訊 → 假設），失敗記「未能測試」 | 已落實（aggregate 在閘門沒過時，把主指標全部覆寫為未能測試） |
| dev／test | 沒有劃分，因為沒有任何參數是用資料調出來的，理由成立。PMI 的 0 門檻是構造出來的，不是調出來的 |
| 成本與重試 | GPU 上限 3 小時。程式錯誤修正不限次，但假設檢定只跑一次 |

## 實作者自己的詮釋
1. **P1 的 FAIL 比 46 原案嚴（上界 < 45 就直接 FAIL，不給 CS-CLIP 救援）**：46 的原文是「CLIP+LLR < 45 **而且** CS-CLIP+LLR 也沒有比 CS-CLIP 高 ≥ 5pt 才停」，也就是兩條路擇一 PASS。46 第 19 行還把「LLR 在 CS-CLIP 上明顯超過 CS-CLIP」當成升到「中」信心的關鍵證據。現行版本會在 CLIP B/32 明確失敗時直接停題，即使 CS-CLIP 那條路大幅成功也一樣。這會**偏向 FAIL**，也改變了題目要回答的問題。請見必修 2。
2. **E2 設計自補**（用 COCO 類別、"a photo of …" 模板、J 恰好缺 1 個條件）：設計本身合理。但我實際模擬後發現**兩個問題**，見必修 4 與建議 1。
3. **PMI 池是 transductive（用的是評測影像本身，沒有用到標籤）**：檢索時影像庫本來就是已知的，這種用法正當；同子集的池也有報出來。我不認為這會明顯偏向 PASS。為了保險，建議 2 加一個外部池的敏感度分析。
4. **用同一流程實測的 CS-CLIP holistic 當救援比較基準，而不是論文的 58.2**：配對比較更乾淨，**同意**。

## 必修
1. **（stage1 前）commit**：repo 目前沒有任何 commit。
2. **（stage1 前，必須由使用者定案）P1 的結構**，二選一，寫進 PREREG 與 `decide_p1`，並更新 unit_tests：
   - (a) **維持現行嚴格版**：在 PREREG 寫明這是刻意比 46 嚴，並經使用者同意。
   - (b) **忠實版（我建議）**：46 的兩條路擇一，按通用規則各用 α/2 = 0.0125 單尾，也就是 1.25／98.75 百分位。
     - PASS：路線 a（CLIP+LLR 下界 ≥ 45）或路線 b（CS-CLIP+LLR − CS-CLIP holistic 的點估計 ≥ 5pp 且下界 > 0）任一成立。
     - FAIL：路線 a 的上界 < 45，**而且**路線 b 的上界 < 5pp。
     - 灰區：其餘情況。HT 資料是固定的，沒有辦法加資料，所以灰區不加碼，直接交使用者。
   - stage1 結束時會跑 aggregate 並印出 P1，所以**必須在 stage1 前定案**，否則等於看了結果才改規則。
3. **（stage1 前）預先快取 OpenAI CLIP 的 `ViT-B-32.pt`**：G1 的官方腳本走 openai-clip 套件，會在 `HOME=icml/tmp/crosscheck/home` 下載這個檔案，但目前沒有快取。計算節點 hgx-cn01 能不能連外網我沒有確認。如果不能，`crosscheck_repo.py --model clip_b32` 會失敗，而 stage1 開了 `set -e`，整個 job 會中斷，連 cs_clip 都不會跑。請在登入節點執行：
   ```bash
   mkdir -p /work/ChihChungHsu/cactus8603/icml/tmp/crosscheck/home/.cache/clip
   HOME=/work/ChihChungHsu/cactus8603/icml/tmp/crosscheck/home PYTHONPATH=/work/ChihChungHsu/cactus8603/icml/envs/t2irare_extra_cscomp \
     /work/ChihChungHsu/cactus8603/icml/envs/t2irare/bin/python -c "import clip; clip.load('ViT-B/32', device='cpu')"
   ```
4. **（stage2 前）E2 的 k_top 實際上只會是 4**：我在 COCO val2017 上，以面積 ≥ 2%、非 crowd 為條件模擬 `e2_build_queries.build`（seed 20261114）：
   - k=4：765 個查詢，來自 268 張影像；
   - k=5：132 個（< 200）；
   - k=6：只有 8 個。
   
   所以 P2 實際上是 err(4) − err(1)，而 46 原案寫的是 k=1…6。另外，765 個查詢只來自 268 張影像，平均每張約 2.9 個查詢，查詢層級的 bootstrap 會**低估 CI**，結果容易偏向「明確」的判定。要做三件事：
   - 在 PREREG 寫明預期 k_top=4（k=6 不可行），並由使用者確認接受；或改用面積 1% 的門檻，但這要先模擬查詢數再定案。
   - `e2_analyze.py` 的 bootstrap 改為**以 I* 影像為單位的群集 bootstrap**，包括 `bootstrap_ci` 與 `delta_err`。
   - 相鄰下降的單調性檢查只看 k=1..k_top，現行程式已是如此。

## 建議（非必修）
1. **P2 幾乎是構造上必然通過**：k=1 時 J 不含 Q 的任何類別（很容易分辨），k=4 時 J 含 4 個中的 3 個。即使是逐條件的聚合器（unit_min、llr_sum），錯誤率也會隨 k 上升，因為其他 k−1 個單元的雜訊會累加。所以 P2 PASS 只是很弱的證據，實際上由 P1 決定生死。建議加一個次要指標「Δ_holistic − Δ_llr_sum」，配對、群集 bootstrap，用來回答聚合器是否比 holistic 少被稀釋。這比較接近命題四真正要主張的內容。
2. P1 加一個敏感度分析：用 COCO val2017 的 5000 張影像當外部 PMI 池，只描述，不判。
3. `bootstrap_ci` 的 α 寫死在程式裡，沒有讀 params 的 `alpha_per_primary`。改 params 時程式不會跟著變，建議改成從 params 讀。
4. 46 提到的 CORE reranker 旗標（COR ≥ 90%）無法判定，已列為未解風險。這件事要在最終報告中向使用者交代。

## 我直接修了什麼
1. `scripts/aggregate.py`：整體判定改成「任一主指標 FAIL → 停損」優先於「未能測試」。原本先檢查未能測試，會讓「P1 FAIL + P2 資料不足」顯示成未能測試，與 PREREG §5 不一致。閘門沒過時兩個主指標都已被覆寫為未能測試，所以不受這個修改影響。
2. `slurm/*.sbatch`：`--output` 改為絕對路徑。
3. `PREREG.md`：新增修訂紀錄。

## 環境與資源檢查
- 使用 `envs/t2irare` 加上 `t2irare_extra_cscomp`，都存在，沒有引用 neon。
- Half-Truths、CS-CLIP、timm 版 OpenAI CLIP 已在 HF 快取中。
- COCO val2017 已解壓（5000 張）。
- unit_tests 先前已全數通過（`results/gates/g1_unit.json`）。
- README「尚未下載／尚未解壓」的敘述已過時，建議更新。
- 資源：1 GPU，各 job 上限 20 分鐘或 1 小時，足夠；磁碟 < 6GB。

## 送出順序（第一個該送的 job：`bench.sbatch`）
```bash
cd /work/ChihChungHsu/cactus8603/icml27-topic-search/killtests/compositional-retrieval
# 必修 2（P1 定案）、必修 3（快取 ViT-B-32.pt）→ commit（必修 1）
sbatch slurm/bench.sbatch                        # 10 分鐘吞吐實測
sbatch slurm/stage1_ht.sbatch                    # G1–G3 + E0／E1；看 results/aggregate.md 的閘門與 P1
# 閘門全過且必修 4 完成、commit 後
sbatch slurm/stage2_e2.sbatch
```
