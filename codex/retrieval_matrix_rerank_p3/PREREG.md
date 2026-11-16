# [codex] P3 固定重排 backend 接線

[codex] 2026-11-16（台北）。此批僅準備程式及CPU合成驗收，沒有P3模型推論或效果。正式plan須經root獨立審查、commit與乾淨工作樹；第一階段及重排模型的真實P0證據不齊時，連GPU smoke都不得執行。

第一階段固定兩個不同模型，一個MLLM embedding、一個CLIP，僅依development證據選定後凍結，不看COLA或其他test結果挑模型。完整P2 order與其圖庫／查詢／來源收據不可更換。每題從原full order取前200，不使用positives、ignore或hard_list選候選，不補K。主效果K100，K50/200深度分析先限定conjunction-dev與negation-dev；本批不接MCMR的native top100，因它不提供完整order／K200。

首批只接已釘版CORE2/8 released Swift backend。每個重排模型均需完整COLA／SCPP／NegBench MCQ報告、至少兩個不同benchmark在1pp內，並有非產生者獨立重算收據綁原arrays及report bytes。重排的完整spec須等於MCQ已核spec；raw score及原request模板不更換。Qwen/Jina/通用MLLM後續須另外固定官方MCQ與適配規格，不借CORE的P0資格。

正式pair batch4、engine max_batch32，沿已核CORE request：原query、assistant image、system `Determine if the following image is relevant to the query. `；fp16、image1200tokens、context1500、原activated FP32分數，不補零、不裁切、不重新縮放。圖文僅image-only document；商品metadata融合不屬本批。

每組以K200一次產分數並保存原始raw response，對K50/100/200的共同prefix使用同一分數，避免數值／padding差異污染深度比較。每題實際評分200 distinct pairs；成本只能稱K200共享cache成本，不冒稱K100或K50獨立實測延遲。若日後需要獨立K100服務延遲，必須另凍結技術計時批。重排平手沿原embedding序，prefix外tail完全保留；只在CPU計分移除ignore，不補候選。

storage chunk64內每次實際backend batch仍4，保存原pair ID、查詢、image SHA及原score/raw。64可被4整除，storage分段不改實際batch邊界。每4對模型回傳先保存immutable raw transaction，checkpoint64提交array；中斷恢復時重驗原raw及模型／候選identity，直接恢復既存分數。已保存raw不再呼叫模型，未保存的呼叫可重算，其STARTED／FAILED／COMPLETE journal與實際pair次數、時間及peak保留。GPU只產分數與技術smoke，CPU分析另行執行，invocation先記STARTED，再於完成／可處理失敗寫wall time，GPU peak從所有已留呼叫收據取最大。硬殺可留下無end receipt的呼叫／invocation，其未知時間與peak明示，不當作零成本。GPU入口只接受root中央dispatcher的owner/task/visible設定且恰一張卡。
