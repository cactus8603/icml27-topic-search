# Rebuttal 升分案例與寫法

三篇 ICLR 2027 案例來自 11/06 21:54 與 11/14 20:12（台北時間）的 Radar 快照。公開討論串用來核對評審原問題、作者的回答，以及評審對回覆的評價。當屆舊評語版本須通過瀏覽器驗證，本筆記以整篇分數分布計算升幅，個別評審僅列目前分數或其明說的改分。歷史案例若有明確的初末分，另標出是哪一位評審。 [Radar 固定版本](https://github.com/dipta007/iclr-2027-rebuttal-radar/tree/a67aefa2c4c2d231e471abab4c36cb771c672ec9/docs/data) · [ICLR 2027 官方時程](https://iclr.cc/Conferences/2027/Dates)

## ICLR 2027 原始討論

### #18295 OpenART，2.00 → 3.33

| 評審問了什麼 | 作者怎麼答 | 評審後續 |
|---|---|---|
| **Lg4s** 質疑能否推廣到外部 benchmark、結果是否因隨機種子變動、人工稽核怎麼做，也要看比較成本。 | 在三個外部 benchmark 補測，用五個種子報標準差與 95% CI，另報基線成本。人工稽核寫清楚分母、抽樣與通過規則。評估器稽核 1,000 組，兩個測試都對才算通過，共 993 組。另有一組針對最終結果的人工盲審。 | 說自己的問題已解決，表示要加分，仍想看作者如何回答另一位評審的維護疑慮。現行總分 3。 [原評語](https://openreview.net/forum?id=ZPk8kVArJP&noteId=SN15xNCzt4) · [作者回覆](https://openreview.net/forum?id=ZPk8kVArJP&noteId=JaI0kbYjtd) · [稽核說明](https://openreview.net/forum?id=ZPk8kVArJP&noteId=9nSJmhOjCJ) · [評審跟進](https://openreview.net/forum?id=ZPk8kVArJP&noteId=WViFFSDzXC) |
| **Dq3z** 認為方法章難懂，要求一個例子貫穿各階段，明列輸入輸出，也希望釐清評估器驗證。 | 實際重寫三節，用同一個每週報告的例子串起流程，補評估器如何用已知標籤做單元測試，將難讀的大表改成兩張熱圖。 | 回覆說疑慮已處理，並在原評語註明升到 4（clear accept）。 [原評語及更新](https://openreview.net/forum?id=ZPk8kVArJP&noteId=aKhRhZeq72) · [作者回覆](https://openreview.net/forum?id=ZPk8kVArJP&noteId=5Jcijhj4TS) · [評審跟進](https://openreview.net/forum?id=ZPk8kVArJP&noteId=jcsQXcnLuR) |
| **zPdR** 問 99.3% 到底量了什麼、人工判斷有沒有受到自動分數影響，以及不同系統日後如何維護轉接。 | 拆開評估器的通過率和最終結果的人工盲審，補上各自分母和錯誤率。說明跑實驗的模型介面與適用範圍。 | 指出詳細答覆與補實驗解決大部分疑慮，升到 3。系統介面日後變動時的維護與擴充仍是其最大疑慮。 [原評語](https://openreview.net/forum?id=ZPk8kVArJP&noteId=En9UjXHmhm) · [作者回覆](https://openreview.net/forum?id=ZPk8kVArJP&noteId=XDlm3tlOPK) · [評審跟進](https://openreview.net/forum?id=ZPk8kVArJP&noteId=1kufJIsMXb) |

同一個結果可以有兩種審稿障礙。一位不確定方法如何執行，作者重寫流程就能直接解除。另一位不確定指標可信度，需要把評估單位、盲審和誤判拆開交代。兩種回覆各自對應實際擋分點。

### #30643 Feature Extraction Advantage of Graph Transformers，2.20 → 3.20

- **問法。** Hnou 指出證明是固定 dictionary 的 proximal update，正文卻套到每層都換 filter；group sparsity 用行或列也前後不一致。sgGd 追問最低重建風險取決於真實訊號統計量，這究竟是 oracle 比較還是模型從 noisy data 就能學到的表現？各模型的輸入、架構和限制是否相同？ [Hnou 原評語](https://openreview.net/forum?id=UjLT8kHce8&noteId=0ngU4IThkf) · [sgGd 原評語](https://openreview.net/forum?id=UjLT8kHce8&noteId=LR1s0Mtj9B)
- **答法。** 作者把「任意逐層變動仍是同一個 solver」收斂為可以證明的兩階段構造，每階段固定自己的 dictionary，證明收斂到兩階段解的組合。統一為 columnwise sparsity，寫清有噪音的假設、無噪音邊界，以及 population oracle 所需的資訊。作者也把共享基底下的權重改善與換基底帶來的額外改善拆開。 [修訂與回答](https://openreview.net/forum?id=UjLT8kHce8&noteId=a2F58Z4xwX) · [跟進回答](https://openreview.net/forum?id=UjLT8kHce8&noteId=QFIqqURo7G) · [oracle 回覆](https://openreview.net/forum?id=UjLT8kHce8&noteId=vkfQ863NeT)
- **改分理由。** Hnou 點名定理範圍和稀疏性的解釋已釐清，表明升分。sgGd 明說疑慮已解，升分。兩位目前均為 4。另位評審 8nxP 肯定作者補的 960 組受控條件，認為較能說清楚 bound 有多緊，仍維持 3，對研究動機與重要性保留。 [Hnou 跟進](https://openreview.net/forum?id=UjLT8kHce8&noteId=zYTmRobH27) · [sgGd 跟進](https://openreview.net/forum?id=UjLT8kHce8&noteId=pjdc4Cz1cP) · [維持分數的理由](https://openreview.net/forum?id=UjLT8kHce8&noteId=7EWf5Pso33)

### #28436 Learning without Overwriting，2.33 → 3.33

- **問法。** 6kE7 質疑原實驗從一個人工構造的圖開始，沒有說明這跟實際 LLM 推理行為的關係，也問兩項理論假設究竟差在哪裡。v2TF 直接說，若能以真實資料做 OPSD 對 SFT 的受控遺忘實驗，會提高評價。 [6kE7 原評語](https://openreview.net/forum?id=zQnQL2xLJ9&noteId=i2LEpnEHqA) · [v2TF 原評語](https://openreview.net/forum?id=zQnQL2xLJ9&noteId=HquZlKCptE)
- **答法。** 從 Qwen3-1.7B 在 MATH 的推理軌跡建圖，以同一個代數訓練後的模型接著學機率，四個種子、16 步更新。自蒸餾的新任務分數提高 25.18 個百分點，舊代數提高 0.54 點；SFT 分別是 +14.22、−23.74 點。作者另說明假設 3 只放寬一個邊界狀態的 coverage，並界定結論仍是受控的圖上續學。 [實驗回覆](https://openreview.net/forum?id=zQnQL2xLJ9&noteId=z1N7r9u8YV) · [理論範圍回覆](https://openreview.net/forum?id=zQnQL2xLJ9&noteId=TsvOo39WuM)
- **改分理由。** 6kE7 說回覆釐清了研究範圍，已更新分數，目前為 3。v2TF 目前為 4，其原評語已事先點名想看的實驗。公開討論中沒有找到 v2TF 的改分說明。 [6kE7 跟進](https://openreview.net/forum?id=zQnQL2xLJ9&noteId=2SyVtQCjnk)

## 歷史對照

- **先按評審指定的預算重跑。** ICLR 2025〈Physics of Language Models Part 2.2〉的 ZAr2 質疑只比 LoRA，欠缺短程全參數 fine-tuning。作者補跑原預算的十分之一，按該預算重設 learning-rate decay，比四種 retry rate，並寫清 i-GSM 合成資料的限制。評審先維持，翌日升到 **8（原 6）**，明說看重實驗的嚴謹度和限制的透明度。 [原評語](https://openreview.net/forum?id=zpDGwcmMV4&noteId=owpeaEDMSn) · [補實驗](https://openreview.net/forum?id=zpDGwcmMV4&noteId=9L62Hj1D1k) · [加分說明](https://openreview.net/forum?id=zpDGwcmMV4&noteId=p2xwKEJdZX) · [初末分數封存](https://github.com/papercopilot/iclr-insights/blob/main/dataset/iclr2025_threshold6_11677_reviewers.json)
- **做一個會讓簡單法贏的對照。** NeurIPS 2020〈Learning to Utilize Shaping Rewards〉被問單一常數權重是否就夠。作者真的測了這個簡單 baseline，在 reward 一律有益或一律有害時，單一權重勝出。在有些狀態有益、另一些有害時，自適應權重才勝出。Reviewer 1 點名新圖回答了疑問，**6 → 7**，仍要求作者釐清另一個從零開始的 baseline。 [作者回覆](https://proceedings.neurips.cc/paper_files/paper/2020/file/b710915795b9e9c02cf10d6d2bdb688c-AuthorFeedback.pdf) · [審稿與改分](https://proceedings.neurips.cc/paper_files/paper/2020/file/b710915795b9e9c02cf10d6d2bdb688c-Review.html)
- **範圍過大的 claim 要改文字。** ICLR 2024〈Large Language Models Cannot Self-Correct Reasoning Yet〉一位評審看了 GPT-4、中性 prompt 與 temperature 0 的補測，只把 contribution 子分 **2 → 3**。作者提出較窄的標題後，評審才有條件地把總分 **3 → 6**。OpenReview 刊載的最終標題仍是原標題。另一位評審因新增模型與實驗結果明說 **6 → 8**。 [原評語與更新](https://openreview.net/forum?id=IkmD3fKBPQ&noteId=b2CsLfx0dr) · [作者跟進](https://openreview.net/forum?id=IkmD3fKBPQ&noteId=pNtK6ZAy7p) · [另一位評審的改分](https://openreview.net/forum?id=IkmD3fKBPQ&noteId=DFtNcvk87n)

## 落到自己的回覆

1. 把每位評審的主要顧慮改成一句可回答的問題。比如「SOTA 比較不足」要具體到哪個 baseline、哪個預算和哪個指標。先處理總分理由或跨評審重複提出的問題。
2. 補能區分兩種解釋的最小實驗。同一資料、預算和評估設定下，直接測 reviewer 提的替代方法或失效子集。寫出效應量、多少樣本，以及適用範圍。若結果有勝有負，保留兩邊。
3. 理論或 claim 過頭時，說清楚證明涵蓋什麼、修正哪一句話、邊界如何處理。用新版頁碼或表號讓 reviewer 和 AC 直接找到證據。
4. 追問時用三句說清楚原疑慮、已新增的結果、仍可能未解的部分，讓 reviewer 可以確認釐清的是同一件事。

ICLR 2024–2025 的觀察研究中，有證據的澄清與升分正相關，初始分數和其他 reviewer 的評分也是重要預測變數。這些案例能示範回覆方式，抽樣自有公開對話的個案，無法估算任何寫法的因果效果。 [研究 §4.2](https://arxiv.org/html/2511.15462v1#S4.SS2)
