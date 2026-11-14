# PREREG：多條件檢索的單調證據聚合（回報 46 的 E0–E2 停損實驗）

狀態：草稿，待 Opus 審查後 commit 凍結。凍結後的修改須另記日期並標為「探索性」。
數值參數全部在 `configs/params.json`（隨本文件凍結）；判讀由 `scripts/aggregate.py` 機械執行。

## 0. 假設與範圍
假設：對單向量檢索，加入正確細節應使分數上升、加入錯誤細節應使分數下降；以對數概似比（PMI）加總的聚合器能同時做到，且 holistic 相似度的錯誤會隨條件數 k 增加。
只跑 E0–E2，純推論，不訓練。成本上限：GPU 3 小時（預估 <1 小時），磁碟 40 GB。

## 1. 資料、模型、seed
- Half-Truths（HF `kbora/Half-Truths`，CC-BY-4.0）三子集共 1437 列：coco_qwen 509、coco_mistral 454、cc3m_qwen 474。全部列都用，不過濾。
- 模型：OpenAI CLIP ViT-B/32（`clip_b32`）、CS-CLIP B/32（`cs_clip`，HF `kbora/CS-CLIP`，MIT）。推論 fp32、無隨機性。
- E2：COCO val2017 全部 5000 張影像與 instances 標註。
- 無 dev／test 劃分：沒有任何參數是用資料調的（見 §3 的所有規則都是事先固定的）。
- seed：bootstrap 20261114（10000 次）；E2 查詢 20261114，E2 加碼查詢 20261115。
- 可重試：程式錯誤修正不限次（記錄於 log.md）；一旦閘門 G1–G3 通過，假設檢定只執行一次，不得因結果不佳調整規則重跑。重跑與失敗全部保留。

## 2. 閘門（失敗 = 該題「未能測試」，限 3 個工作天修；與題目不成立分開記）
- **G1 程式正確性**：(a) `unit_tests.py` 全過（COR 手算、PMI 手算與性質、LLR 的 anchor 抵銷恆等式、細節抽取、bootstrap、E2 查詢建構與分析、判讀函式邊界）；(b) 與官方 `evaluate_half_truth.py` 逐列 cosine 最大絕對差 <= 1e-4（兩模型、coco_qwen）；(c) 細節抽取覆蓋率（anchor token 完整保留且兩個細節皆非空）>= 95%（每模型每子集）。
- **G2 重現**（阻擋）：CS-CLIP 的 holistic COR 在三子集皆落在 58.2／50.4／42.2 的 ±3pp。診斷（不阻擋，偏離須解釋）：CLIP B/32 COR 對 18.9／19.2／24.5、HT 準確率對官方 README（CLIP 34.0／28.6／36.1；CS-CLIP 76.4／67.2／66.2）。
- **G3 雜訊底線**：(a) 各 holistic COR 的 bootstrap 95% 區間半寬 <= 5pp；(b) 同一輸入以不同 batch size 重跑，所有 cosine 最大絕對差 <= 1e-5。

## 3. 評分定義（固定）
單元：anchor 一個單元；新增細節一個單元。細節 = 以小寫 token 的 difflib 比對，取延伸句相對 anchor 被插入或替換的字詞（`common.extract_detail`），不看影像與模型分數。
- holistic：整句 cosine（官方評分）。
- llr_sum（**主聚合器**）：score(T) = Σ_單元 PMI_c(I)，PMI_c(I) = τ·s(I,c) − log mean_{I'∈池} exp(τ·s(I',c))，τ = 模型 logit_scale.exp()，池 = 三子集所有唯一影像（依 sha256 去重，1437 列內的唯一影像，不使用任何正確性標籤）。A 只含 anchor 單元；A+ 含 [anchor, d+]；A− 含 [anchor, d−]。因 anchor 項相同，COR = P(PMI(d+)>0>PMI(d−))（恆等式，已單元測試）。
- 比較用（描述，不參與判準）：unit_mean、unit_min、unit_sum、llr_sum_samesubset（池只用同子集）。
- COR：嚴格 s(A+) > s(A) > s(A−)，平手算錯。

## 4. 統計與多重性
- 有 2 個主指標（P1、P2），整體 α = 0.05 以 Bonferroni 除以 2。**P2 單尾 α = 0.025**（bootstrap 百分位 2.5%／97.5%）。**P1 為兩條路徑擇一，各用單尾 α = 0.0125**（百分位 1.25%／98.75%，輸出欄位 `lo_1s`／`hi_1s`），2026-11-14 審查後定案。
- P1 的兩條路徑都是主判準的一部分（還原 46 原案），不再是「灰區加碼」。
- 列層級 bootstrap（HT）；E2 為**以 I* 影像為單位的群集 bootstrap**（同一影像的查詢一併重抽；Δ 以 I* 影像聯集為群集，k_top 與 k=1 共用同一抽樣）。

## 5. 主指標與三態（回報 46 的門檻改寫）

### P1（E1 主停損）
（2026-11-14 審查後、結果前定案，非探索性：還原 46 原案的兩條路擇一）
- 路線 a：CLIP B/32 + llr_sum 在 COCO(Qwen) 的 COR，單尾 α=0.0125 下界（1.25 百分位）>= 45。
- 路線 b：CS-CLIP + llr_sum 的 COCO(Qwen) COR 比同流程 CS-CLIP holistic（實測值，非論文 58.2）高：點估計 >= 5pp，且配對 bootstrap 單尾 α=0.0125 下界 > 0。
- PASS：路線 a 或 b 任一成立。
- FAIL：路線 a 的上界（98.75 百分位）< 45，**且**路線 b 的上界 < 5pp（兩條路都明確失敗）。
- 灰區：其餘。HT 資料固定，無法加資料，**灰區不加碼**，直接交使用者。
- 實作：`configs/params.json: p1.path_a / path_b / alpha_one_sided_per_path`、`aggregate.py: decide_p1`、`analyze_ht.py`（輸出 lo_1s／hi_1s）三處已同步，單元測試涵蓋兩條路徑。

### P2（E2 條件數稀釋）
指標：**CLIP B/32 holistic 的 pairwise 錯誤率差 Δ = err(k_top) − err(k=1)（pp）**。err(k) = 1 − P(s(I*,Q) > s(J,Q))，I* 含 Q 全部 k 個條件，J 恰缺其中 1 個（平手算錯）；**k_top = 4**（2026-11-14 定案：k_list = [1,2,3,4]；COCO val2017、面積>=2%、非 crowd 下實查 k=4 有 765 個查詢（來自 268 張影像）、k=5 只有 132、k=6 只有 8，不足 200，**接受**；若 k=4 的查詢數 < 200 則視為未能測試）。
- PASS：Δ 下界 >= 5pp，且 k=1..k_top 的錯誤率序列相鄰下降最大 <= 2pp（單調性，容差 2pp）。
- FAIL：Δ 上界 < 5pp。
- 灰區：其餘。**唯一加碼**：合併第二組 seed（20261115）的同規模查詢，重算一次；仍灰區則維持灰區，交使用者。
條件定義：COCO 類別，影像中至少一個非 crowd 實例且面積 >= 影像 2% 才算存在；每 k 最多 1000 查詢（k=4 預期約 765），Q 為 I* 的類別隨機 k 子集，J 由符合者隨機取一；文字模板 "a photo of a, b and c"（類別依 id 排序）。

### 整體
任一主指標 FAIL → 停損；兩者 PASS → 繼續；其餘（含灰區）→ 交使用者。任一閘門失敗 → 未能測試。

## 6. 非判準的記錄項
- E2 次要（2026-11-14 新增）：holistic 與 llr_sum 的稀釋差 Δ_holistic − Δ_llr_sum（Δ = err(k_top) − err(1)，同一批查詢配對、以 I* 影像群集 bootstrap；>0 表示 holistic 比逐條件聚合器更被條件數稀釋）。輸出於 `analysis_*.json: secondary_dilution`，**只描述，不參與任何判定**。
- E2 次要：各聚合器（unit_mean／min、llr_sum）的 pairwise 正確率與 recall@50（定義：top-50 命中相關影像數 / min(50, 相關影像數)；相關 = 含 Q 全部類別）。
- E0：兩模型三子集的 COR／HT（overall、entity、relation）；foil_type 分項；條目為描述。
- CORE reranker：未取得可用程式（模型卡的 GitHub 連結為空、需 transformers>=4.57 與自有 wrapper），**本次未評估**；46 的「reranker COR >= 90% 則限定固定索引情境」旗標無法判定，列為未解風險。

## 7. 已知設計限制（事先聲明）
- 細節抽取是機械規則，對 subject_object_swap 類（binding）細節會截斷（例如 "refrigerator is laying on top of"），LLR 預期對此類無優勢；不調整規則。
- PMI 池使用評測影像本身（無標籤、transductive）；same-subset 池為探索性對照。
- 細節單元去脈絡：LLR 對 anchor 的貢獻抵銷，主指標等於「細節自身的 PMI 符號檢定」，其 0 門檻是構造而來的，不是調出來的。

## 修訂紀錄
- 2026-11-14（Opus 審查，機械修正，非判準變更）：`scripts/aggregate.py` 整體判定改為「任一主指標 FAIL → 停損」優先於另一主指標的「未能測試」（與 §5 文字一致）；slurm 輸出路徑改為絕對路徑。判準層面的意見見 `REVIEW.md`，尚待使用者定案。
- 2026-11-14（審查後、結果前定案，非探索性；協調者決定，依 REVIEW.md 必修 2–4，任何假設結果出來之前）：
  1. P1 改回原案的雙路徑擇一（見 §5），各用單尾 α=0.0125；`params.json`、`aggregate.py`、本檔同步。取代「明確失敗不救援」的嚴格版。
  2. G1 官方腳本交叉核對所需的 OpenAI CLIP `ViT-B-32.pt` 已快取於 `/work/ChihChungHsu/cactus8603/icml/tmp/crosscheck/home/.cache/clip/ViT-B-32.pt`（`crosscheck_repo.py` 以 HOME=該 scratch/home 呼叫官方腳本，clip 預設讀 `~/.cache/clip`；已在登入節點以同樣 HOME 與 PYTHONPATH 驗證 `clip.load('ViT-B/32')` 成功，檔案存在時不會連網；sha256 由 clip 載入時自動校驗）。計算節點不必連外網。
  3. E2：k_top = 4（接受 k=5、6 查詢數不足）；bootstrap 改為以 I* 影像為單位的群集 bootstrap（含 Δ）；以合成資料單元測試。
  4. 新增次要指標：holistic 與 llr_sum 的稀釋差（不參與判定）。
