# [codex] CORE MCQ 首批獨立送件前核對

[codex] 2026-11-16 21:02（台北）。root 為此模組非實作者；只檢查官方來源、原始資料與CPU技術證據，未產生或讀取本批模型效果，未開自建test queries。

結論：CORE-Reranker-2B／8B 的完整MCQ首批可按固定設定送入共享3090佇列；這是送件前協定驗收，不是GPU smoke或論文數字通過。獨立審查後的plan SHA256為 `276ee0e88ded86f41b539df437ac4c718dac329be6d054f3702379e5594e3c8e`。

- 官方 `Alibaba-NLP/core-emb@400cf919838ceeecf4c94e3ac178a4270abcc4c5` 的COLA、SCPP、NegBench逐分支核對：CORE的COLA是每圖選caption兩個嚴格比較；SCPP兩個正caption皆須勝負caption；NegBench caption0須勝其餘三，平手失敗。指令尾端空白按各官方分支保留。NegBench COCO／VOC與SCPP各子集只在各benchmark內等權宏平均。
- root獨立從釘版原始Parquet／JSON／CSV重建15,912個原題與58,891個pairs，與凍結清冊逐欄、順序完全一致。9,425張圖片的實際SHA／bytes全驗；重複題與原有效分母保留。
- root另逐檔驗兩組CORE實際權重、478個Swift Python來源的SHA；10項CPU合成測試獨立重跑通過。純CPU processor證據顯示1280×960合成圖用1200 image tokens、總1271 tokens，沒有執行GPU模型。
- 模板由作者checkpoint `args.json`自動選`qwen3_vl_reranker`，ms-swift官方generative reranker取最後有效位置yes/no logits，fp32 sigmoid後原始`content[0]`保存；不用既有的自製評分器。fp16、FlashAttention2、context1500、image token1200與官方發布程式對齊。
- 單次呼叫4pairs與原engine max_batch32分開記；先以合成資料驗batch1／4差≤.01，失敗不自動換batch／精度／權重／offload。GPU僅輸出原score，主結果待明確CPU分析及非實作者獨立重算。

已知差異仍保留：checkpoint記Swift4.0.0.dev0，公開可取得環境固定4.0.0；不能宣稱精確重建未公開的development commit。送件前只更正來源清冊一處repo名稱為`core-emb`，更新其plan SHA，未改評分或資料規則。原始失敗不以零分靜默填補。

此首批只含CORE2／8，其他已下載Qwen／Jina／通用MLLM需另立協定與凍結批次，不因存在backend class就當已通過P0或已送件。
