# 三個 rebuttal 衍生題目的深入調查

2026 年 11 月 14 日。**三題都保留，先補各題缺的證據。** 驗證器與解碼題已重算公開資料。Attention gate 題已有原論文與近作的對照，強 baseline 下的剩餘現象需要量。Astra 已複核三題的文獻重疊和實驗設計。

## 三個驗證階段

1. **現象在不在？** 先在公平對照、留出資料和有競爭力的 baseline 下重現。量大小、發生頻率與適用範圍。可把「觀察到失敗」和「知道它為什麼發生」記成兩個結果。
2. **訊號夠不夠強？目前方法還能改善多少？** 先量訊號在留出資料有沒有超過便宜替代訊號的預測力，再測用它改變決策是否真的改善結果。用更多標註或計算可先量潛力。若診斷允許看真值或未來結果，這只是在該資訊條件下的參考上限。在訓練集擬合的策略於留出集的表現，則是該策略類別的可達估計。明列資訊、候選範圍及成本，最後與重調過的強 baseline 比同等成本下的品質或錯誤率。樣本不足、量測失真、訊號太弱、強 baseline 已接近可達結果，各自需要不同的下一步。
3. **重新設計並逐次驗證。** 先做能隔離機制的介入，再用簡單方法測收益能否重現，接著壓低成本和擴大適用範圍。每輪保留配對控制，記錄回收了多少差距，以及品質、時間或標註成本變在哪裡。第一版效果小時，先定位是訊號取得太貴、近似漏掉關鍵情況，還是介入的位置或時機不對。

| 方向 | 一、現象 | 二、訊號與改善空間 | 三、方法候選 |
|---|---|---|---|
| 強 verifier 有誤時的線上驗證 | 公開答案已核實兩模型共同誤收。專家標註的自然語言證明待核實。 | 已有完整標註下的留出成本估算。下一步遮住多數金標，扣除初始校準、持續抽查和模型成本，與同樣可問強模型的固定級聯及均勻抽查比。 | 先保留均勻探索與直送專家，再依可觀測的風險訊號分配複核。逐步比較雙向錯誤率及總成本。 |
| 大型 commit set 的聯合交互作用 | 大 block 掉分與局部非加成各有證據，兩者的因果關係待量。 | 固定解碼狀態、候選 token 值、提交量和續跑規則，比精確成對與精確聯合重排，量最終答對的差。另與調過的簡單交換和 SetCommit 比。 | 先試校正成對加總，再試只針對大集合、接近門檻的位置做額外聯合探測。把探測和續跑都算進延遲。 |
| Attention gate 與 optimizer | 原作在高學習率有差異。保留要問的現象是使用 QK norm 與調好的 optimizer 後，attention 寫入突變是否仍可重現且影響驗證 loss。加 sandwich norm 作強對照。 | 比較寫入突變、QK logits 和更新幅度能否預測下一批 loss。從同一 checkpoint 比定向修正與同等更新幅度的整體縮步，查可回收的差距。 | 從少數層、少數步的訓練期修正開始。驗證正常排程下能否更快達到相同 validation NLL，以及延長訓練後效益是否保留。 |

先從可用 CPU 重播的驗證器開始，並把解碼同狀態對照和 gate 的強 baseline 成對試驗寫清楚。這是建議的驗證順序，三題的實驗結果各自記錄。

## 強 verifier 與少量抽查

ICML 2026 Spotlight〈When to Trust the Cheap Check〉的 reviewer 指出，當強模型判錯，錯誤會跟著進到線上門檻更新。正式版控制相對於強模型的風險。近期 AM-PPI 已在共同成本下分配多個代理及專家標註，NeurIPS 2024 Active RCPS 已做線上風險校準與主動標註，OMVV 和 Selective Auditing 已做以正確 gold 為基準的線上決策。可檢驗的範圍縮到**會出錯的付費驗證資訊能否改變當下的雙向判決，並在只見少量 gold 時維持實際成本與真實誤收、誤拒的前緣**。 [ICML 評語](https://openreview.net/forum?id=dfN4HMZbc9&noteId=NmvsiMRa3E) · [AM-PPI](https://arxiv.org/html/2605.08429) · [Active RCPS](https://arxiv.org/abs/2406.10490) · [Selective Auditing](https://arxiv.org/html/2610.31662v1)

在公開的 1,869 個候選答案、147 個獨立題目叢集，Qwen3-32B 對 758 個錯解誤收 114 個，其中 101 個 Qwen3-0.6B 也收下。按題目分五折，假設問強模型成本為一次 gold 的 10%，用訓練折完整金標估設定，測試折平均成本從 **0.619 降至 0.285**，兩類錯誤的點估計都約 9%。換 Llama-3.1-70B-Instruct，類似條件為 **0.619 → 0.543**。這是公開答案資料的部署潛力，尚未計入取得訓練金標的成本。專家標註的自然語言證明有 888 筆，但需要另外取得成對 verdict 或信心分數。 [題目深查](2026-11-14-deepcheck-fallible-verification.md) · [成本分析與重算](2026-11-14-deepcheck-verifier-replay-data.md)

## 大型平行解碼集合

重算 Measured Coupling 的 1,319 題逐題檔，同 K、同門檻，LLaDA-8B-Instruct 在 GSM8K 的 block 20 為 **76.27%**、block 64 為 **67.70%**，配對差 **8.57 個百分點**。原作的 **15.6%** 是 block 20、一次提交至少八個 token 時，剩餘 masked 位置的門檻分歧。block 64 同類統計是 **10.8%**，因此沒有根據把整批 8.57 點的落差歸給非加成誤差。SetCommit 已做固定提交量、候選與續跑規則的同狀態比較。報告留下精確成對和精確聯合的配對分叉，只有看到額外收益後才量含探測成本的真實延遲。 [Measured Coupling 附錄 H、J、K](https://arxiv.org/html/2611.08307v1) · [SetCommit](https://arxiv.org/html/2610.31120) · [題目深查](2026-11-14-deepcheck-large-commit-sets.md)

## Attention gate 與 optimizer

〈Gated Attention〉原論文 48 層、1.7B、400B tokens 的高 LR 實驗，PPL 為 **9.195（baseline）／7.407（sandwich norm）／7.325（elementwise gate）**。相對原 baseline 的 PPL 改善約 96% 已能由 sandwich norm 取得。Qwen3.8 的另一種架構在相同結構改用 Muon 後，2× LR 的每萬步 loss spikes 是 **4.3 → 0.2**，加入其 gated residual 後同樣是 0.2，正常學習率的生產訓練仍另有 gating 收益。若後續要測，只保留固定輸入下限制 optimizer 一步造成的 attention 寫入變化，並以同驗證品質所需的真實時間為主指標。 [Gated Attention 原文](https://arxiv.org/html/2505.06708) · [Qwen3.8 §3.3](https://arxiv.org/html/2608.30320#S3.SS3) · [題目深查](2026-11-14-deepcheck-gate-optimizer.md)
