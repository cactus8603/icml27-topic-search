S2 題目研究規劃整理（唯讀，沒有改任何檔案）

讀過的範圍：README.md、AMD.md、CLAUDE.md、cards/ 全部 10 張、go-topics/ 全部（2 篇正文、3 篇 context、索引；3 張 png 與 make_S2.py 只看了檔名與用途）、paper/ 全部的實質文字（draft.md、appendix.md、paper.tex、main.tex、appendix.tex、tables/*.tex、evidence.json、citation-sources.json、template-source.json、README.md、build.sh）。ICML 官方 .sty／.bst／template_original 沒讀。另跑了 `git log --stat` 和兩個關鍵 commit 的 diff。為了釐清規劃，也看了 results/README.md、baselines/README.md、configs/next-comparison.json、results/baseline-suite/development-report.md 開頭和 slurm/run-h200.slurm 開頭。下文 `路徑:行號` 是文件原文；標【推論】的是我自己的判斷。

---

## 0. 演進脈絡（git log）

- 11-11 `f92e037`→`9642cbd`：W1 設計卡、程式，結果 GO。
- 11-13 `ac42b5a`→`0b0bd09`：W2 設計，結果 NO-GO（P3），同一個 commit 一起改了兩項 §3a 檢查。
- 11-13 15:29 `08f2233`：W3 設計卡與程式（未開跑）。run 目錄時間戳是 `20261113-152918`，比 commit 晚 11 秒。
- `51478f7`（另一條分支）：改正 W1 sink 數字的歸屬，再由 `e680541` 合併。
- `43d55e6`：收錄 W3、B1、RULER DEV、資料、上游 go-topics。
- 11-14 `b4457fe`：121 檔、+51,678 行。加入 baseline 與 repair 實作、近期方法清單、六設定計畫、H200 handoff。
- `57ee800`：新增 CLAUDE.md。

整體走向（文件自己的說法）：題目從「gate 拿掉 sink → 保護 sink 的收益消失」的**分析題**，先轉成「gate × value 淘汰分數（gval）」的**方法題**，再轉成「從強 baseline 保留集合出發的受限集合修補（repair）」。README 已經改名為「KV Cache Compression Research」，並說 S2／W／B 代號「只保留在歷史紀錄」（README.md:1-3）。

## 1. 核心想法：方法的定義與公式

### 共同設定
- 三個配對 checkpoint：`QwQZh/gated_attention` @ `aad415c`，分別是 baseline、gate_headwise、gate_elementwise。約 1.72B 參數，28 層，16 個 Q head，8 個 KV head（W1.md:75-77；draft.md:20）。
- **Gate 的定義**：`o_{h,q} = g_{h,q} ⊙ Σ_k a_{h,q,k} v_{j(h),k}`。headwise 時 g 是純量，elementwise 時是向量（draft.md:22-28）。實作上就是 SDPA 輸出乘 `sigmoid(gate_score)`（W1.md:99）。
- 不是乾淨的「只差 gate」對照：elementwise 的 FFN intermediate 從 6144 改成 5504（W1.md:77）。論文草稿因此明說這「not a gate-only causal intervention」（draft.md:20）。

### 第一階段：sink 分析（W1／W2）
- **sink 質量**：對首 token 的注意力質量，取 query 位置 ≥16 的平均（W1.md:61）。
- 三種保留策略：S0 只留最近 W 個；S4 留前 4 個加最近 W−4 個。
- 收益 Δ = NLL(S0) − NLL(S4)，只算位置 ≥W 的 token。
- 縮減率 R = 1 − Δ_gated／Δ_baseline（W1.md:10-16）。

### 第二階段：gval 淘汰分數（B1／W3）
- 公式：`s_{h,k} = (1/n_k) Σ_{q≥k} a_{h,q,k} ‖g_{h,q} ⊙ v_{j(h),k}‖²`，其中 n_k = C−k+1；再在同一個 KV group 內平均（draft.md:50-62；W3.md:10）。
- baseline 模型沒有 gate，改用「隱式 gate」1 − a_{q,0}（W3.md:37）。
- 控制組：
  - oval：gate 設為 1
  - gval_shuffled：打亂 query 與 gate 的對應
  - gobs：拿掉 value 項
  - gval_win64/pool5：只用最後 64 個 query，再做寬 5 的 pooling（W3.md:36-44；draft.md:66）
- 保留規則：所有方法都固定保留首 1 個與最近 128 個前文 key，計入預算 W（B1.md:11）。
- 協定 B：前文 3584 個 token 一次完整 prefill，之後只壓縮前文的 KV，再續寫 512 個 token（W2.md:14）。

### 第三階段（目前的方向）：受限集合修補（repair）
文件：S2-Astra-method-design.md、S2-repair-implementation.md、S2-repair-exploration.md。

- 對單層、單一 KV group，給定保留集合 S，定義：
  - `m = Σ_{k∈S} a`
  - `u = Σ_{k∈S} a·v`
  - `r = o − u/m`
- 精確的局部誤差：`e_exact = ‖Σ_{h∈H(c)} W_O^h diag(g_qh) r_qh‖²`
- 便宜的對角代理：`ℓ_q = (1/c_q) Σ_{h,d} b_hd g_qhd² r_qhd²`，其中 b_hd = ‖W_O^h[:,d]‖²
- 目標函數：最後 32 個 prefill query 中，取損失最大的 8 個求平均（TopMean，等同 CVaR）（impl.md:7-53）。
- 搜尋程序：從 pinned SnapKV 的保留集合 S0 出發。用梯度挑候選，每步最多試 4×4 對交換，最多兩步；要求嚴格下降，且相對下降 ≥1e-4（impl.md:73-100；Astra.md:75-80）。
- 六個變體（exploration.md:15-30）：
  - Mean：identity gate、平均誤差
  - Tail：identity gate、尾部誤差
  - Tail+Gate：native gate、尾部誤差
  - Projected：直接用 Wo 投影的誤差 `‖Σ_h W_h(g⊙r)‖²/c_q`
  - Entropic：τ=0.5 的 logmeanexp
  - Mass-trust：要求保留的 mass ≥ 0.9 倍初始值

### 一段話版本（照文件，可對審稿人複述）
Softmax 注意力會把多餘的質量倒到首 token（sink）。StreamingLLM、H2O、SnapKV 這類方法隱含地依賴 sink。這個專案用已釋出的配對 1.7B 模型（無 gate／headwise gate／elementwise gate），在固定 prompt-KV 預算 W 下研究兩件事：

1. gate 消除 sink 之後，保護前幾個 token 的收益會不會消失？會，R 約 0.93–0.99。
2. 原生 output gate 能不能當成 KV 淘汰的訊號？先試了 gval（attention × gate 後的 value 能量）。它在 PG-19 的壓縮額外 NLL 上贏過 Observed，但在 RULER 落後 CAOTE 11.48 個百分點。

目前改成另一個方向：從強 baseline 的保留集合出發，在固定 query、logits、value、gate 的條件下，用「gate 加權、經 Wo 投影的局部輸出誤差」的尾部風險當目標，只做少量交換。文件自己說「三項貢獻目前都未成立」（Astra.md:115），論文草稿也說目前「not a successful compression method claim」（draft.md:134）。

## 2. 論文打算主張的貢獻與狀態

論文草稿**沒有明確的貢獻清單**。標題是「Preliminary Experiments and Evaluation Setup」（main.tex:29），草稿說最終貢獻「not yet fixed」（draft.md:14）。下表是我從各卡整理出來的。

| 候選主張 | 狀態 | 出處 |
|---|---|---|
| gate 讓保護前 4 個 token 的收益縮減 93–99.5%；對打分式策略、T=16384 也成立 | **已有證據**：只有 1.7B、PG-19、NLL | W1.md:118-121；W2.md:207-212 |
| 逐 head 機制：sink 質量能預測哪些 head 需要保護（原本「獨立成文的核心」） | **失敗、撤回**（ρ=0.265） | W2.md:206,213；research-notes.md:80 |
| gval 降低壓縮額外 NLL | **部分**：只有 headwise GO，而且用的是重用過的書；elementwise 待判定 | W3.md:76-88 |
| 改善下游任務 | **未成立**（落後 11.48pp） | W4.md:146 |
| query-specific gate 提供額外訊號 | W3 的 NLL 層級有支持（H3），下游**未驗證**；repair 中 Tail+Gate 沒有高過 Tail | W3.md:83-84；exploration.md:145-148 |
| 尾部失效機制、repair 方法的新穎性、品質／成本收益 | **都未成立**（文件原話） | Astra.md:115 |
| repair 在 DEV208 的品質 | 探索結果：Projected 37.98 對 SnapKV 37.31，區間 [−1.23, 2.66] 跨 0 | exploration.md:149-152 |
| benchmark／baseline 基礎設施（978 次相容性檢查、凍結 DEV 矩陣） | **部分**：matrix 完成 4/8 | B2.md:31；development-report.md |
| 效率（延遲、記憶體） | **尚未開始**；現有 runner 不能拿來量效率 | README.md:87；W4.md:121-125 |
| 第二個模型家族或尺寸（Qwen3.5） | **尚未開始** | W4.md:23,127 |
| 獨立確認（PG-19 validation、RULER reserved、LongBench reserved） | **尚未開始** | W4.md:3,148 |

## 3. 各階段的目的、門檻原文與結果

### W1（11-11，H200，之後 AMD 重跑）
- **目的**：gate 拿掉 sink 之後，保護前 4 個 token 的收益是否縮小（W1.md:6）。
- **門檻原文**：「GO：P0、P1 成立，而且 R ≥ 0.30。R 的 95% CI 下界也 ≥ 0.30 才算穩」（W1.md:26）。
  - P0：baseline sink ≥30%，兩個 gated 模型 ≤10%
  - P1：baseline Δ>0，CI 不含 0
  - P2：R ≥0.30（W1.md:20-22）
- **結果**：GO。sink 35.3%／1.9%／0.4%；留出集 W=256 的 R = 0.967 [0.966, 0.969]、0.933 [0.920, 0.943]（W1.md:118-124）。AMD 結果一致（AMD.md:28）。
- **要注意的地方**：
  - 判 GO 時，§3a 六項檢查全部沒打勾，§9 也有三項沒打勾（W1.md:31-37, 96-100）。詳細版已點出這點（S2-詳細.md:310）。
  - **sink 數字的更正（`51478f7`）**：35.3%→1.9%／0.4% 原本寫在「留出 W=256」底下，其實是前 20 段、開發與留出混合的平均（S2-詳細.md:306,339）。commit 訊息寫「卡片在 §11 追記」，但這個 commit 實際只改了 README.md 一行。W1 卡的 §11 至今**沒有**這則追記（W1.md:118 只寫「全層平均」）。後來這個更正寫進了 README.md:115、go-topics/README.md:25、paper/appendix.md:11。`e680541` 就是合併這個更正。判定不受影響。
  - 原 W1 的 GO 門檻（R ≥0.30）來自 playbook 的「縮小至少 30%」（S2-詳細.md:84）。
  - 文件自承的效度風險：「sink 沒了、保護沒用」可能被審稿人視為顯然（S2.md:51）。

### W2（11-13，3090）
- **目的**：做三件事。P3 每個 head 的 Spearman；P4 換成打分式策略（observed／snapkv）；P5 T=16384 並用 S4r（W2.md:6-9）。
- **門檻原文**：
  - P3：「ρ 和 CI 下界都 ≥ 0.50，判 GO；CI 上界 < 0.50，判 NO-GO」
  - 「整體 NO-GO：任一項 NO-GO」
  - 「P3 NO-GO：拿掉 fig1 機制圖，只報整體現象」
  - 「P3、P4a、P4b 三項都 NO-GO：題目降級」（W2.md:30-40）
- **結果**：整體 NO-GO。
  - P3：ρ=0.265 [0.238, 0.278]
  - P4a：R_obs = 0.997／0.995，GO
  - P4b：R_snap = 0.993／0.989，GO
  - P5：0.962／0.917，GO（W2.md:205-213）
- **事後修改**：
  - smoke 失敗後改了 §3a 兩項檢查的量法：policies 改用 fp32；perhead 的雜訊改量「同批次」。卡上寫「門檻不變、開跑前」（W2.md:47,49）。
  - 這些修改和結果在同一個 commit `0b0bd09` 一起提交，§2／§3 沒有改（我比對過 diff）。
  - 正式 run 的 anchor 與 longctx 用 `ac42b5a`，perhead 與 policies 用修補後的程式（W2.md:199）。
- **計畫外的發現**（文件自己標為探索）：gated 模型**沒有比較好壓縮**。各自用最佳策略時打平，或 gated 差 0–0.03 nat；協定 B 下 +0.008／+0.012（W2.md:221-232）。
- **W2 給的下一步**：做 gate 加權 observed 的先導；「無效的話，題目降級為分析短文或 workshop 論文」（W2.md:234-236）。

### B1（探索，沒有 GO 判準）
- 結果：DEV 20 本／39 段。gobs−observed @256 為 −0.00630（headwise）、−0.00571（elementwise）。gobs−CAOTE 跨 0（B1.md:64）。
- 舊 pilot 的數字比較大，但保留規則和精度設定不同，沒有混入主表（B1.md:58）。

### W3（11-13，3090）
- **目的**：在留出集預先註冊確認 gval（W3.md:6）。
- **門檻原文**：
  - 「GO：gate_headwise 的 H1、H2、H3 都成立」
  - H1：gval−observed 的 CI 上界 <0，且改善 ≥5%
  - H2：對 CAOTE、FastCAOTE 的 CI 上界都 <0
  - H3：對 oval、shuffled 的 CI 上界都 <0
  - 都用 98.33% CI（W3.md:14-23）
- **結果**：headwise GO（改善 12.6%）；elementwise 待判定（H2／H3 跨 0）（W3.md:76-86）。
- **要注意的地方**：
  - 卡片 §1 說這是「沒看過的留出集」（W3.md:6），但這 49 本書已經在 W2 policies 用過。§11 事後承認「不能稱完全獨立確認」（W3.md:88）。
  - §3a 四項檢查都沒打勾（W3.md:30-33）。
  - W3.md:26 寫「不論結果，下一步都是 Qwen3.5 的下游任務」，但實際下一步是 1.7B 的 RULER。
  - 12.6% 指的是壓縮額外 NLL 減少的比例，不是總 NLL 或準確率（W3.md:88）。

### RULER DEV ／ W4 D0（11-13）
- **門檻原文**：「候選 DEV macro 比該參照高 ≥2 個百分點，才使用 C2 新資料」（W4.md:55）。
- **結果**：headwise W=1024 下，gval 28.75、gval_win 34.45、SnapKV 41.80、CAOTE 45.94。候選落後 11.48pp，**未達門檻，C2 不啟動**（W4.md:146）。
- README 總表把這輪標成「待判定」（README.md:119），理由是 D0 還缺 FastCAOTE、RocketKV、controls。

### W4（C0／C1／D0／C2／E1／E2／E3）
- 狀態：**只有設計與輸入準備，模型確認實驗都還沒跑**（W4.md:3,148）。

| 階段 | 做什麼 | 門檻或狀態 |
|---|---|---|
| C0 | 實作檢查 | — |
| C1 | 50 本新 PG-19 書 | GO＝五項 99% CI 上界都 <0，且改善 ≥5%（W4.md:39） |
| C2 | RULER 512 題 | 候選−參照的 98.75% CI 下界 >0，且 ≥+2pp（W4.md:64） |
| E1 | LongBench 614 筆 | — |
| E2 | 成本 | — |
| E3 | Qwen3.5 0.8B／2B | — |

- 文件規定 E 系列不能拿來取代失敗的 C2（W4.md:25）。
- 精度規劃：50 本書大約只能偵測 0.00873 nat 的效應；如果真實效應只有 5%，約需 190 本（W4.md:43）。

### B2／B3（基礎建設，明確「不是研究 GO」）
- B2：978 次相容性 smoke 全部通過（B2.md:31）。
- B3：凍結 DEV 矩陣，共 59,168 次評估。規定 `ready_for_method_design` 必須為 true 才能設計新方法（B3.md:12,27）。
- 目前進度：Full KV 8/8，matrix 4/8（development-report.md）。

### Repair 探索（11-14，DEV208，探索性）
Astra 卡事先寫了停止規則（Astra.md:103）：
- +2pp 只是「繼續投入門檻」
- 「B 不優於 A 則尾部機制未受支持」
- 「C 不優於 B 則放棄 gate 敘事」

實際結果（exploration.md:143-152,186）：

| 設定 | DEV208 macro |
|---|---|
| Full KV | 91.43 |
| SnapKV | 37.31 |
| CAOTE | 28.14 |
| Mean（A） | 37.75 |
| Tail（B） | 37.65 |
| Tail+Gate（C） | 37.72 |
| Projected | 37.98 |
| Mass-trust | 37.17 |
| Entropic | 37.13 |

- 五個候選相對 SnapKV 的區間都跨 0。
- 8 個 NIAH 任務上，Full 98.63，SnapKV 與 Projected 都只有 18.75（exploration.md:154）。
- 每個 group 平均只淨換入約 1.125 個 token（exploration.md:156）。
- 舊矩陣（不同 runner）的參考值：Afterglow+SnapKV 89.30、KeyDiff 69.01、DropKV 61.61、ValueDiff 58.76（exploration.md:158,186）。
- 【推論】照 Astra 卡自己的規則：B−A = −0.10，所以尾部機制不受支持；C−B = +0.07，遠低於 +2pp，等於該「放棄 gate 敘事」。但 exploration 卡沒有正式套用這條規則下判定，只寫「gate、projection、mass constraint 的額外收益尚未被確立」（exploration.md:152）。

## 4. 文件自己承認的弱點、風險與待辦

- **新穎性**：
  - 集合誤差公式已見 CAOTE Appendix B（impl.md:21）。
  - 投影式誤差已有 CriticalKV（impl.md:38）。
  - 「精確誤差＋貪婪交換」已有 Trust the Mass／ContourKV（exploration.md:207）。
  - GateSight 在其受測模型上 gate 幾乎不改變 head 內排序（Astra.md:29）。
  - 「尚未找到完全相同的設計，不等於證明新穎」（Astra.md:33）。
- **效果**：下游沒有恢復核心 retrieval 能力（exploration.md:154）。gval 的 NLL 改善沒有轉成下游改善（draft.md:90）。
- **資料重用與選擇偏差**：
  - W3 用的書重用了（W3.md:88）。
  - smallDEV52 從已看過的 DEV208 抽樣，不是獨立資料（next-comparison.json:129）。
- **因果**：checkpoint 之間不是只差 gate（draft.md:20）。HF 權重對應論文哪一個 1.7B 訓練設定不明（W1.md:95）。
- **規模**：只有一組 1.7B、原生 4K context。Qwen3-8B 只是 baseline 軌，不能拿來驗證 gate 方法（draft.md:30,128）。
- **效率**：沒有任何效率證據。品質 runner 同時保有 full cache，不能量效率（README.md:87；draft.md:130）。eager 模式仍有 T×T attention，不是免平方的實作（impl.md:131）。
- **協定**：屬於「問題已知的單次 prefill 壓縮」，不能稱為線上淘汰或跨 query 重用（W4.md:69）。
- **跨 runner 不可比**：舊矩陣用 BF16 batch4，新候選用 serial batch1；同一個 SnapKV 在兩者分別是 36.66 和 37.31（exploration.md:158）。
- **待辦清單**：
  - 建 smallDEV52
  - 實作 Afterglow 與 KeyDiff 初始化的 adapter（README.md:10）
  - 補兩項診斷：固定支持集比較誤差；initializer × swaps {0,2,16}（exploration.md:164）
  - C0、C1、C2 的 runner
  - 完成 B3 矩陣
  - 確認 ICML 2027 的投稿格式（template-source.json）
  - 讀完 OBCache、CriticalKV 等最近鄰（exploration.md:217）

## 5. 模型、資料、指標、baseline 與 H200 交接

### 模型
- **已用**：
  - 1.7B 三個變體
  - Qwen3-8B（官方 scorer 軌：KVzap、FastKVzip 權重）（baselines/README.md:9-12）
- **只是計畫**：
  - Qwen3.5 0.8B／2B（混合架構，需要新環境與 adapter）（W4.md:23,127）
  - 更早的文件提過 Qwen3.5 9B／35B 和 Qwen3-Next（research-notes.md:68），現在已不在計畫裡

### 資料
- **已用**：
  - PG-19 test 100 本（W1、W2、W3、B1）
  - 舊 RULER 4 任務 × 32 題
  - RULER13 DEV208
  - LongBench 6 任務 DEV136（B3 矩陣中）
- **已準備但未評估**：
  - PG-19 validation 50 本／100 段
  - RULER C2 的 512 題（已有 128 題，另 384 題待生成）（W4.md:60）
  - LongBench reserved 614 筆
  - SCBench（暫不納入）（README.md:139-148；W4.md:129）

### 指標
- PG-19：NLL、壓縮額外 NLL。
- RULER：`string_match_all` 的任務等權 macro。
- LongBench：官方各任務 metric。
- 統計：以書或文件群組為單位的配對 bootstrap 10,000 次。
- 效率（E2，計畫）：prefill、scoring、decode 分開計時；TTFT；峰值記憶體。

### Baseline
- **1.7B 軌（14 個設定）**：Full、random、prefix/recent、value norm、Observed、SnapKV、TOVA、Knorm、KeyDiff、CAOTE、FastCAOTE、ValueDiff、Afterglow+SnapKV、DropKV。
- **Qwen3 軌額外有**：Expected Attention、PyramidKV、AdaKV、CriticalKV、KVzip、KVzap、FastKVzip（baselines/README.md:46-55）。
- **未接入**：RocketKV、IndexMem++、GateSight（GateSight 的 repo 回 404）（B2.md:35）。
- **下一輪收斂成六個設定**：F（Full）、S（SnapKV）、A0／A1（Afterglow±Projected）、K0／K1（KeyDiff±Projected）。主要比較是 A1−A0、K1−K0（baselines/README.md:103-112）。狀態是「planned_not_executable」（next-comparison.json:136）。
- **finalist 階段才加**：DropKV、Mean。Compactor 是唯一的新機制候補。

### H200 交接
- 分工：H200 跑「整理並凍結後的完整匹配比較」；3090 只做 smoke 和小 DEV（README.md:29；baselines/README.md:18-23）。
- H200 上現在**只能跑**五設定 baseline（Full、SnapKV、Afterglow+SnapKV、KeyDiff、DropKV，W1024），「**沒有包含尚未實作的四個初始化／repair 配對**」（README.md:62）。
- 步驟（README.md:31-82）：
  1. clone repo，執行 `setup-workspace.sh`
  2. 把 DEV208 小包（348,231 bytes）搬到 `incoming/`，執行驗證與解包
  3. `sbatch --check`
  4. 每個任務跑一題 smoke
  5. 用 `--time=04:00:00` 送出完整 DEV208
- 約束：
  - 不接寫 3090 的半份 run（README.md:8）
  - 兩個平台合計最多四個方法工作（README.md:85）
  - H200 上的歷史權重可能已被清掉（README.md:44）
  - H200 的實際耗時要靠 smoke 量測，不從顯示卡規格推算（README.md:29）

## 6. 文件之間的矛盾與含糊處

1. **`51478f7` 的說法和實際不符**：訊息說「卡片在 §11 追記」，實際只改了 README，W1 卡沒有追記。go-topics/S2.md:7,22 仍然沒有限定地寫 35.3%→1.9%／0.4%（雖然頁首有歷史快照的警告）。
2. **「配對可以隔離 gate」的說法前後不一**：早期說可以隔離 gate 效應（S2.md:11；research-notes.md:67），論文草稿說不是 gate-only 對照（draft.md:20）。
3. **W3 的「沒看過的留出集」（W3.md:6）與資料重用的事實互相矛盾**，只在事後承認。
4. **最強參照不一致**：舊 4 任務 RULER 是 CAOTE 45.94 > SnapKV 41.80；DEV208 反過來，CAOTE 28.14 < SnapKV 37.31，而且 Afterglow+SnapKV 89.30 遠高。W4 D0 的參照池沒有 Afterglow（W4.md:53）。
5. **流程閘門被繞過**：B3（B3.md:12,27）、impl（impl.md:153,179）和 Astra（Astra.md:124-125）都規定 baseline 矩陣完成之後才設計或試新方法。但 repair 在矩陣只完成 4/8 時就跑完了四組 DEV208，理由是「使用者授權並行」，並標為「探索性」（exploration.md:3,81）。
6. **論文稿已過時**：
   - 進度表顯示 Full 92/208、matrix 0（tables/baseline-dev-status.tex），實際是 Full 8/8、matrix 4/8。
   - evidence.json 的 `full_completed: 0`。
   - 草稿完全沒提 repair、Afterglow 89.30 和六設定計畫。
   - 草稿說硬體「undecided」（draft.md:132；B2.md:37），但 11-14 的 README 已定 H200。
7. **v1／v2 含糊**：B3 說有效的 run 目錄是 `dev-freeze-20261113-v2`（B3.md:14），但 development-report 和 evidence 的方案 ID 是 `...-v1`。【推論】可能是「freeze ID v1」和「run 目錄 v2」兩件事，但文件沒有說明。
8. **RULER 的「512 題」有兩種意思**：README.md:143 說已收集 512 題（4 任務 × 4K／8K × 2 種子 × 32 題）；README.md:164 和 W4.md:60 說 C2 的 512 題中只有 128 題，另 384 題待生成。
9. **W3 說的下一步沒做**：W3.md:26 說下一步一定是 Qwen3.5 下游任務，實際沒做，baseline 軌改用不帶 gate 的 Qwen3-8B。
10. **W1 引用平台**：論文與 evidence 引用 MI300X 的「reproduction」數字（draft.md:72），但原始結果是 H200，AMD 是之後的重跑（S2-詳細.md:336）。數值本身一致。
11. **判 GO 時檢查清單未勾**：W1 §3a／§9 有未勾項目（W1.md:31-37, 96-100）；W3 §3a 也是空的（W3.md:30-33）。
12. **選 Projected 進下一輪的依據**：Projected 被固定成下一輪的 repair，但文件自己說它「尚未證明優於 Mean」（next-comparison.json:125），區間也跨 0。
13. **外部 API 的使用**：方法設計與程式審查用了外部模型 `gpt-6-astra`（OpenAI API，暫時金鑰）。卡上記載第一次外傳被自動核准拒絕，之後經使用者同意才送出（Astra.md:3,127）。這不是研究矛盾，但涉及把專案內容送到外部服務，值得你知道。

## 7. 做決定前最該親自看的 5 個檔案

1. `/work/ChihChungHsu/cactus8603/icml27-s2-gated-sink-kv/cards/S2-repair-exploration.md`（尤其 :137-221）：最新的實際結果。Projected +0.67pp 跨 0，NIAH 沒恢復，跨 runner 的 Afterglow 是 89.30，還有六設定計畫的由來。目前方向值不值得做，主要看這裡。
2. `/work/ChihChungHsu/cactus8603/icml27-s2-gated-sink-kv/cards/S2-Astra-method-design.md`：repair 方向的動機、哪些部分「不可當作新意」的最近鄰表（:21-33）、A/B/C 的停止規則（:88-109），以及「三項貢獻都未成立」（:115）。用它對照第 1 個檔案的數字，就能判斷 gate 敘事是否已觸發停止。
3. `/work/ChihChungHsu/cactus8603/icml27-s2-gated-sink-kv/cards/S2-W4.md`：唯一完整預先註冊的確認路線（C1／C2／E1–E3）與 D0 的 −11.48pp 結果。精度規劃（:43）說 50 本書可能不足以確認小效應。
4. `/work/ChihChungHsu/cactus8603/icml27-s2-gated-sink-kv/cards/S2-W2.md`（§11，:196-236）：原題核心 P3 的 NO-GO，以及「gated 模型沒有更好壓縮」的計畫外發現。這決定了原本的分析題還剩多少可以寫。
5. `/work/ChihChungHsu/cactus8603/icml27-s2-gated-sink-kv/baselines/README.md`（:16-25、:90-116），搭配 `configs/next-comparison.json`：H200 交接後實際要跑什麼，哪些仍然是 planned_not_executable，以及以 2025 年起方法為核心對手的方針。

備選：`/work/ChihChungHsu/cactus8603/icml27-s2-gated-sink-kv/paper/draft.md`，可以看目前論文敢主張到哪裡（:12, :126-134）。
