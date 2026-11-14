## 先講結論

**這個題目的核心空缺已經大致被填掉了，「同記憶體」的實用性論述在現有數字下站不住。**

1. 2026 年 9–10 月至少有三篇「量化後模型自己生成軌跡，再由全精度教師逐 token 修正」的論文：QAOPD 2609.26708、OnlineQAT 2610.09346、Right On Q（簡稱 OnQ）2610.29378。這就是切入點 (a)。
2. OnQ 把 Qwen3-8B 2 位元的 MATH-500 拉到 90.60（全精度 95.53，差 4.9 分）。ThinkBit（2610.34886）在 2.03 bpw 下做到 92.2（全精度 96.9）。所以 MATH-500 已經回到 5 分以內。AIME 還差 24–28 分，LiveCodeBench 還差約 19 分。
3. 同記憶體比較的結論是 2 位元大模型會輸。單看 OnQ 自己的表，3 位元 4B 只用 GPTQ（純 PTQ，幾分鐘就能做完），AIME 就有 51.85，高於 2 位元 8B 的 OnQ 44.07。而且 3 位元 4B 佔的記憶體還比較小。

**讀取深度說明（每項都適用）：** 標「讀全文」的，是用網頁抓取工具讀 arXiv HTML 全文，由抽取模型整理表格；超過 10 萬字元的頁面只讀到前 10 萬字（主表都在這段內），我沒有逐字親讀。Okoshi 那篇我下載 PDF 轉成文字後，親自讀了表格原文。標「只讀摘要」的，是用 arXiv API 讀摘要。

---

## 1. 指定論文總表

| 論文 | 讀取深度 | 模型 | 位元 | 類別 | 全精度 → 量化後（重點） | 成本 | 程式／權重 |
|---|---|---|---|---|---|---|---|
| 2601.14888 Reasoning-QAT | 讀全文 | R1-Qwen-1.5B、Qwen3-0.6B/4B | W3/W2 G128、W4A4KV4 | QAT（知識蒸餾＋GRPO） | 4B W2：AIME-120 58.89→22.78、MATH 95.33→78.27、LCB 48.38→0.00（與你核對的一致）。BitDistiller、EfficientQAT 只在 1.5B W3 比過 | 沒寫 GPU 小時。LAQuant 測得「ReasoningQAT」在 8B 上要 78.44 H200 小時（見下方注意） | 文中沒有連結 |
| 2606.02011 失敗模式＋救援 | 讀全文 | Qwen3-8B/32B | W2A16、W4A16，用社群 GPTQ 現成權重 | 解碼時補救 | 32B：MATH 93.0→65.0（規劃＋救援 87.2）、AIME26 70.0→16.67（40.0）。8B：MATH 92.4→17.2（救援 74.2）、GPQA 59.09→17.17。W4A16 的準確率沒有給 | 評測約 100 A100 小時。退回 FP16 的比例沒有報 | github.com/brain-lab-research/quantized-reasoning（有程式與 results 資料夾） |
| 2608.01078 ScaleQ-1.58／AYOT | 讀全文 | Qwen3 1.7B–235B、R1-Llama-70B | W1.58A16 | PTQ（CAT-Q＋全精度模型自己生成的軌跡當校準資料） | MATH-500：4B 96.8→58.4、8B 96.8→63.0、14B 97.2→76.8、32B 96.8→85.2。沒有 AIME、GPQA、LCB | 8×A100 單機跑 4–240 小時（依模型大小） | 寫「將釋出」於 github.com/IntelChina-AI/BitTern（repo 存在，只看了目錄，沒確認程式是否完整） |
| Okoshi 等 ICLR 2026 | **親讀 PDF 表格** | Qwen3-1.7B/4B/8B | W3/W2 G128 | QAT（分塊校準＋教師導引的獎勵修正損失） | 見下方補充表 | 1.7B W2：第一步約 4、微調約 80 H100 小時。GRPO 對照組要 190–1016 小時 | github.com/yasu0001/ReasoningQAT（基於 EfficientQAT，沒有權重，也沒有 arXiv 版） |
| 2602.02581 QuantLRM | 讀全文 | R1-Llama-70B、R1-Qwen-32B、R1-Qwen3-8B、Olmo-3-7B | 只有 W3、W4，**沒有 2 位元** | PTQ | R1-Qwen3-8B W3：AIME-120 59.2→40.0（GPTQ 8.3、AWQ 34.2）。沒有 MATH-500、LCB | 類似 AWQ，幾分鐘 | github.com/psunlpgroup/QuantLRM |
| 2509.12464 RAC | 讀全文 | R1-Distill、Qwen3 | **是剪枝，不是量化** | PTQ 剪枝，用推理過程（CoT）的激活值當校準 | Qwen3-8B 剪 50%：MATH .962→.862、AIME25 .633→.500 | 單張 H100 | github.com/RyanLucas3/Reasoning-Aware-Compression |
| 2504.02010 | 讀全文（前 10 萬字） | R1 671B 與蒸餾版 | 1.58–2.51 動態量化（只在 R1-671B）、3/4 位元 | 評測研究 | R1-Distill-Qwen-7B 3 位元 GPTQ：AIME24 46.7→17.8。明說沒做 2 位元 | — | github.com/psunlpgroup/Compression-Effects |
| 2609.00224 QTEA | 讀全文 | Qwen3 基礎模型、Llama3 | 1.7 bpw 三值 | PTQ | **沒測任何推理任務**，只有困惑度與 7 個零樣本常識題 | 256 筆校準樣本 | github.com/Intelligent-Microsystems-Lab/QTEA |

**Okoshi 的絕對分數補充**（Table 2，G128，依序為 MATH500／LCB／MMLU-Redux／GPQA／IFEval，平均）：

| 模型 | BF16 | 4 位元 GPTQ | 3 位元本法 | 2 位元本法 | 2 位元 GPTQ／AWQ |
|---|---|---|---|---|---|
| Qwen3-1.7B | 89.0/53.6/74.7/38.4/70.4，平均 65.2 | 平均 58.8 | 82.7/33.0/67.7/31.7/61.0，平均 55.2 | 48.6/6.5/40.1/14.5/32.2，平均 28.4 | 平均 4.2／9.5 |
| Qwen3-4B | 93.6/71.2/84.3/51.5/83.6，平均 76.8 | **93.4/66.2/82.1/50.7/81.2，平均 74.7** | 平均 68.3 | 77.1/19.5/61.7/26.9/50.1，平均 47.1 | 平均 5.7／2.4 |
| Qwen3-8B | 94.0/73.0/87.3/61.6/86.5，平均 80.5 | 平均 79.2 | 91.5/60.0/84.5/47.5/78.8，平均 72.5 | **80.4/28.5/72.6/34.5/59.3，平均 55.1** | 平均 4.6／4.0 |

- Okoshi 的 1.7B W2 對照：EfficientQAT 平均 18.3、BitDistiller 15.2。
- 推理長度（Table 9，8B 的 MATH-500 平均 token 數）：16 位元 5513、3 位元 6296、2 位元 9966。答錯的題目在 2 位元下平均 25679 token。
- Okoshi 沒有測 AIME。

**注意：** OnQ 和 LAQuant 表中的「ReasoningQAT」到底是 2601.14888，還是 Okoshi 的 repo（名稱也叫 ReasoningQAT），我沒能確認。OnQ 的校準設定（4096×2048、兩輪）和 Okoshi 完全相同，比較像是 Okoshi 那篇。

---

## 2. 2026 年 6–11 月的新作

查法：Semantic Scholar 引用清單 API（第二輪查詢多次被限流），加上 arXiv API 關鍵字查詢。

**直接處理「低位元量化的推理模型」的論文：**

- **OnQ 2610.29378**（讀全文）：Qwen3-1.7B/4B/8B，W2/W3 G128；量化後模型自己生成軌跡、全精度教師在同一前綴上給下一個 token 的分布（on-policy 蒸餾，動態加權的前向／反向 KL）。
  - 8B W2：AIME24–26 71.85→44.07、MATH 95.53→90.60、GPQA 59.93→43.43、LCB 73.59→55.04，九項平均 81.82→70.68。
  - 最強基線：BitDistiller 平均 60.42；ReasoningQAT MATH 85.27、AIME 28.15。
  - 沒有 GPU 小時，沒有程式連結。
- **ThinkBit 2610.34886**（讀全文）：混合三值／2/3 位元純量量化＋QAT。
  - 8B（2.08GB）：MATH 96.9→92.2、AIME@16 71.2→46.9。
  - 14B（3.74GB）：97.5→94.8、73.6→57.7。
  - **llama.cpp 的 Q2_K（8B 3.28GB）就有 91.1／45.6**，是大多數論文沒放進比較的強基線。
  - 程式要等論文被接受後才釋出。
- **QAOPD 2609.26708**（讀全文）：QAD 之後接 on-policy 蒸餾（含驗證器獎勵與逐 token 教師對數比）。
  - 只用 Qwen3 0.6B–4B 的非思考模式；沒有 AIME、LCB。
  - 4B 在 W1.88 下 MATH-500 從 2.20 升到 36.00。
  - on-policy 蒸餾 57 GPU 小時，前面的 QAD 要 820 GPU 小時。
  - 程式在 github.com/MingZwhy/QAOPD。
- **OnlineQAT 2610.09346**（只讀摘要）：分塊 QAT 後接 on-policy 蒸餾。Qwen3-1.7B 平均：W3 57.28、W2 32.52，只比 ReasoningQAT 多 0.44 分。
- **RECAP 2610.15259**（讀全文）：W4A4KV4，用全精度模型的長軌跡做校準。Qwen3-4B 的 AIME 58.9→48.6。repo 目前只有 README。
- **LAQuant 2605.08755**（讀全文）：只有 W3/W4。Qwen3-4B W3 的 AIME25 63.75→54.69。8B 成本 14.91 H200 小時。
- **SAGE 2610.14745**（讀全文）：2.25 bpw 向量量化。Qwen3.5-4B 思考模式的 MATH-500 93.2→53.6，QTIP 只有 25.8。
- **ReQAT 2606.15682**（FP4）、**ReSET 2606.13233**（NVFP4）：只讀摘要。
- **2606.00206**、**2606.25519**：Token 膨脹、過度思考類分析，3–4 位元，只讀摘要。
- **2607.11317**、**2607.20129**：CUSUM 監控＋重新解碼，只讀摘要。
- **SteerDistill 2611.13410**：代理任務、W4，只讀摘要。

**差距判定：**
- 8B 2 位元的 MATH-500 已經回到 5 分以內（OnQ −4.9、ThinkBit −4.7）。
- AIME 沒有回到 5 分內（−24 到 −28），LiveCodeBench 也沒有（−18.6）。
- 仍沒被補掉的空缺是**長推理難題（AIME、LCB）**，以及 **32B 等級的 2 位元 QAT**。

**漏查風險：** Okoshi 的引用清單因限流沒查到。Semantic Scholar 對 11 月新論文的收錄有延遲。

---

## 3. 強基線在長推理基準上的 2 位元成績

| 方法 | 有沒有在推理模型的長推理基準上報過 2 位元 | 數字 | 程式 | 8B 能否在 ≤15 GPU 小時內做完 |
|---|---|---|---|---|
| EfficientQAT | 有 | OnQ 表 8B W2：AIME 11.48、MATH 75.00、LCB 33.97。ThinkBit 8B：MATH 71.9、AIME@16 10.3。Okoshi 1.7B：MATH 24.8 | OpenGVLab/EfficientQAT（MIT） | 大概可以，需縮小資料量。Okoshi 1.7B 第一步約 4 H100 小時，8B 按比例推估是估計值 |
| BitDistiller | 有 | OnQ 8B W2：AIME 22.59、MATH 79.00、LCB 35.45。ThinkBit 8B：66.4／7.1 | 已搬到 OpenBitSys/BitDistiller | 照 Okoshi 的設定不行：1.7B 微調就要約 78 H100 小時 |
| QTIP | 只有短推理 | SAGE：Qwen3.5-4B 思考模式，16k 預算，MATH-500 25.8。沒找到 AIME 數字 | Cornell-RelaxML/qtip（GPL-3.0） | 不含微調應該可以（憑記憶，未核實）。需要自訂 kernel，評測時可以先反量化成 BF16 |
| AQLM＋PV-tuning | **沒找到** | — | Vahe1994/AQLM | 憑印象 7–8B 的 AQLM 約需一天以上 A100 時間，PV-tuning 更久，可能超過 15 小時（**未核實**） |
| ParetoQ | **沒找到**（2601.14888 只引用，沒有數字） | — | facebookresearch/ParetoQ | 需要數十億 token 的 QAT，大概不行（**未核實**） |
| Any-precision、MatGPTQ、MoBiQuant | **沒找到**長推理數字。MoBiQuant 只測到 GSM8K | — | SNU-ARC/any-precision-llm、IST-DASLab/MatGPTQ、SimWangArizona/MoBiQuant | PTQ 類，應該可以（未核實） |
| GGUF Q2_K／IQ2 | 有（ThinkBit） | 8B Q2_K：MATH 91.1、AIME@16 45.6，但實際 3.28GB，約 3.2 bpw | llama.cpp | 幾分鐘 |
| AutoRound | 有（OnQ） | 8B W2：MATH 34.0、AIME 0 | — | 短時間 |

---

## 4. 同記憶體對照：**2 位元大模型目前是輸的**

**直接比較過的文獻：**
- ThinkBit 附錄：4B Q4_K_M（2.50GB）MATH-500 96.3（greedy）。作者自己寫「4–5 位元的較小模型分數更高」，高過他們的 14B 2 位元。
- Liu 等 2504.04823（讀全文）：3 位元 32B 勝過 BF16 7B。但那是跟 BF16 比，不是跟 4 位元比，不能拿來撐 2 位元。

**同一張表內的比較（OnQ Table 1，全部在 32k 預算下）：**

| 設定 | 估算記憶體 | AIME | MATH | LCB | GPQA |
|---|---|---|---|---|---|
| 8B W2 OnQ | 約 4.4GB（嵌入層／輸出頭保持 BF16）；把頭也壓成 2 位元可降到約 2.1GB | 44.07 | 90.60 | 55.04 | 43.43 |
| 4B W3 純 GPTQ | 約 2.2GB | **51.85** | **93.73** | **56.97** | **43.94** |
| 4B W3 OnQ | 約 2.2GB | 62.59 | 93.80 | 64.55 | 49.16 |

記憶體是我估算的：Qwen3-8B 的嵌入層和輸出頭不共用，約 1.24B 參數，用 BF16 就佔約 2.5GB。ThinkBit 表中 EfficientQAT 8B 標 4.35GB，可以佐證。4B 的 4 位元（G128＋BF16 嵌入）約 2.7GB。LAQuant 的 4B W4：AIME24 70.42、AIME25 62.66。

**32B 對 14B：**
- 2606.02011：32B 2 位元的 MATH 65.0、AIME26 16.67，連 8B FP16 的 92.4 都不如。不過那是較弱的社群 GPTQ 權重。
- ScaleQ：32B 三值 MATH 85.2，低於 4B BF16 的 96.8。
- 沒找到 32B 2 位元 QAT 對 14B 4 位元的直接數字。

**結論：** 依現有數字，「2 位元大模型優於同記憶體的 4 位元小模型」不成立。實用性論述只剩幾條路：(i) 只看頻寬／速度，但 2 位元會生成更多 token，抵銷速度優勢；(ii) 32B 以上的大模型（沒有數據）；(iii) 把嵌入層和輸出頭也壓下去之後的嚴格位元計算。

---

## 5. 兩個切入點是否已有人做

**(a) 用量化後模型自己的軌跡做 on-policy 蒸餾：已被做過，重疊度高。**
- OnQ：量化模型生成、BF16 教師逐 token 給分布，就是 DAgger 精神。
- QAOPD：明確引用 Ross 等 2011 的 DAgger 論文。
- OnlineQAT：反向 KL 版本。
- 另外，SteerDistill 是閉環教師修正（代理任務），2611.11450 研究 on-policy 蒸餾能容忍多少訓練與推論不一致。
- 剩下的空間只有細節，例如在 AIME 等級的難題上、32B 的規模、計算量更省的做法。

**(b) 解碼中依不確定性或迴圈偵測，只把少數 token 升到較高位元：部分重疊，沒找到完全相同的組合。**
- DMR 2602.02711（讀全文）：同一模型的巢狀精度（W3 對 W4/W8），以「步」為單位路由，共用高精度 KV 快取。只測代理任務，有 22–35% 的 token 在高精度。沒有 2 位元，也沒有長推理。
- 2606.02011：偵測到迴圈就用 FP16 整段重跑，是以整題為單位。
- MoBiQuant：每個 token 選精度的路由器，只測到 GSM8K。
- 2607.20129、2607.11317：監控後回退重新解碼，但精度不變。
- QSpec、QuantSpec：低精度起草、高精度驗證（憑記憶，未重讀）。

**這個方向的主要邏輯問題：** 要隨時升精度，高位元部分就得常駐在記憶體裡，那麼記憶體佔用等於高精度模型，2 位元在記憶體上的理由就消失了，只剩頻寬優勢。要讓它成立，必須論證高位元部分放在 CPU 或快閃記憶體、只偶爾讀取的情境。

---

## 6. 機制解釋做到的程度

**已有的發現：**
- 推理鏈變長、迴圈、跑滿預算、答案出現了卻停不下來（2606.02011 的相關係數；Okoshi Table 9；QAOPD：W2.79 的 0.6B 有 95% 跑滿預算、70% 重複）。
- 推理沒有收尾（ThinkBit：1.7B 只有 69.6% 收尾）。
- 誤差依位置累積（RECAP：只用 4K 窗口校準時，後段 KL 從 0.14 升到 7.33）。
- 量化前後分布 KL 高的位置也是熵高的位置，量化模型在這些位置偏好 "wait" 一類的標記（2606.00206，3–4 位元）。
- 低熵的符號 token（數字、運算子）容易被抽錯（ReQAT、ReSET，FP4）。
- 最後幾層的 MLP 特別敏感（2504.02010）；某些層對精度特別敏感（2611.11450）。
- 端到端損失會淡化 KV 誤差的梯度（LAQuant 的論點）。

**還沒人回答、可以檢驗的問題：**
1. **2 位元**下（不是 3–4 位元）分歧從哪一個 token 開始：教師強制下的逐 token KL，和自由生成時第一次偏離的位置之間是什麼關係。
2. 交換實驗：只在少數 token 換成全精度能救回多少？「關鍵 token」占比多少？
3. 誤差主要來自被污染的 KV 快取（歷史），還是當下這一步的計算？做法是用全精度重算 KV、但當下用 2 位元，反過來再做一次，分開比較。
4. 依層、依模組（注意力／MLP、嵌入／輸出頭）在**自由生成**下的貢獻。
5. on-policy 蒸餾到底修了什麼：是停止行為，還是推理本身？OnQ 只報告重複率下降。
6. 為什麼 MATH-500 能救回，AIME 卻救不回：難度與長度各占多少？

這些問題可以撐起一篇分析論文，但要做成「機制分析加方法」的完整研究，方法部分會直接撞上 OnQ、ThinkBit。

---

## 7. 停損實驗所需資產

- **權重：** Qwen3 1.7B–32B、R1-Distill 在 HF 上都有公開權重（常識，這次沒逐一點開）。
- **量化工具：** GPTQ 用 GPTQModel（ModelCloud，11/14 仍在更新）；AWQ 用 llm-compressor（vllm-project）。AutoAWQ 已停止維護（憑記憶）。
- **評測：** evalscope（Okoshi 和 OnQ 都用它）、LiveCodeBench 官方 repo、ruikangliu/Quantized-Reasoning-Models（MIT，含評測與量化權重）、brain-lab 的 repo（有 vLLM 版本需求檔和 results 資料夾）。
- **用 vLLM 跑 fake-quant 評測：** 可行。把反量化後的 BF16 權重存成一般模型就能直接載入，速度和 BF16 相同。Okoshi 的 repo 有 vLLM 轉換腳本，OnQ 在 verl＋vLLM 上做 rollout。BF16 的捨入會有極小差異。
- **評測時間（估計，不是實測）：**
  - Qwen3-8B 在單張 H200 上，一次 MATH-500（1 次取樣）加 AIME（30 題 ×8 次取樣），32k 預算。
  - 2 位元模型的輸出約 10k（MATH）到 20k（AIME）token，總共約 1000 萬 token。
  - 解碼主要受 KV 快取的讀取頻寬限制，吞吐量估計每秒 2–4k token。
  - 合計每個設定約 **1–3 H200 小時**，32B 約 4 倍。
  - 旁證：2606.02011 報告兩個模型的完整評測約 100 A100 小時；B200 上 8B W4A16 在 batch 8 時約每秒 1312 token。

---

## 8. 總評

**新穎性風險：高。**
- 切入點 (a) 已被 OnQ、QAOPD、OnlineQAT 三篇做過。
- 機制層面的停止、迴圈、長度已有多篇描述。
- 巢狀精度加步級路由已有 DMR。
- 剩下 (b) 的「token 級、2 位元、長推理」組合還沒人做，但有上面講的記憶體邏輯問題。

**可行性風險：中。**
- 資產和評測都齊全，4 張 H200 足夠做 8B。
- 但要在 10 週內超過 OnQ（8B W2 平均 70.68），需要先重現它。OnQ 沒有公開程式，也沒有 GPU 小時；QAOPD 的計算成本很高。

**實用性風險：高。** 同記憶體下輸給 4 位元小模型，這點審稿人很容易提出來。

**10 週內最可能成立的論文形狀：**
- **最穩的：** 一篇嚴格的「記憶體與 token 成本相等」的評估加機制分析。主張是「目前的 2 位元推理模型在同記憶體、同 token 預算下輸給 3–4 位元的小模型；失敗集中在 X（可以用交換實驗定位）」，再附一個輕量、針對該機制的修正。
- **其次：** 把 token 級升精度在「高位元部分放在主機記憶體或快閃記憶體」的設定下做成，量測升精度比例與 AIME 回升的取捨。必須誠實面對頻寬成本。
- **不建議：** 再提一個 on-policy 蒸餾或 QAT 方法正面硬拼。

**沒能確認的事項：**
1. Okoshi 的引用清單（Semantic Scholar 限流）；11 月以後的新論文可能有漏。
2. OnQ、LAQuant 中「ReasoningQAT」到底指哪一篇。
3. OnQ 的程式、權重與訓練 GPU 小時。
4. AQLM＋PV-tuning、ParetoQ、QTIP、any-precision 的 8B 2 位元耗時（我寫的是記憶，未核實）。
5. QSpec、QuantSpec 的細節（憑記憶）。
6. BitTern 的程式是否完整。
7. ThinkBit 的 AIME@16 和 OnQ 的 AIME 24–26 年份與設定不同，不能直接互比。
8. 各模型記憶體 GB 是我依參數量估算的，ThinkBit 和 EfficientQAT 那幾列除外（論文有標）。
9. 評測時間是估計值。
10. 2606.02011 的 FP16 退回比例沒有報。

Okoshi 的 PDF 文字檔在 /tmp/claude-1047/-work-ChihChungHsu-cactus8603-icml27-s2-gated-sink-kv/bd65f9a7-961a-4b8d-adce-42df28fbb74b/scratchpad/okoshi.txt，引用清單的 JSON 在 /tmp/claude-1047/-work-ChihChungHsu-cactus8603-icml27-s2-gated-sink-kv/bd65f9a7-961a-4b8d-adce-42df28fbb74b/scratchpad/cit/。
