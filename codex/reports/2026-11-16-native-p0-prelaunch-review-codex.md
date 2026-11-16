# [codex] SigLIP2 native P0 獨立事前審查

[codex] 2026-11-16 22:04（台北）。實作者 matrix_runner，最終獨立審查 root。程式、資料與技術資格檢查通過，可凍結並送固定官方驗證；尚無 GPU 模型數字，P0 效能狀態仍為 NOT_RUN。沒有讀取主結果。

最終 plan SHA `c62a50a646938173a1fa76221b7181fd53fe037abcb56248f41db0304d44b329`，13 個執行來源的當前 SHA 全部吻合。root 重跑 native 14 項與 CLIP 邊界9項 CPU 測試，共23項 PASS；shell 語法與程式來源亦已查核。最終技術 audit SHA `eb1528bf16a02a5d03d6c813bcfce681d9a1d10a8c616d4bef9eaadaed6f5e60`，綁此版 plan、2,766 個依賴 payload、全部9項實際 module path／module version／distribution version，CUDA 未初始化。

root 另從原始 Karpathy JSON 自行重建 test image order、所有 raw caption／sentence ID、caption→image 對應，與 adapter 輸出逐項比較；再獨立重算全部6,000張圖片 SHA與bytes，全部通過。COCO保留5,000圖／25,010句（4,990張5句、10張6句）；Flickr保留1,000圖／5,000句，零排除。證據在 `dl/retrieval-matrix-native-p0/independent-prelaunch-review.json`。COCO2014 logical檔名與實體2017共用快取的symlink分開核對，不以resolve後basename錯拒相同原圖。

模型使用相同釘版 Google 官方 HF FP32 類別與processor，完整權重、lowercase、長64、原定image4/text32，無量化或offload。模型程式 SHA `bd0c1c78d1f11a5d4b026de94d732212311ce1aede1f07090fb0baf3949caf0f`。純色 CPU forward曾於先前 plan SHA `2be4bbd185b6c80523c7608b999099aebea6ff06b637dc826ae5c6efb9eb6ef4` 通過，模型程式未再變動；此收據僅背書相同模型 bytes 的技術 forward，不冒充目前plan完整執行。GPU前另有固定batch1對照smoke，失敗保留。

producer 計分直接載釘版 Google BigVision scorer（來源 revision／URL／SHA 均保存）；independent scorer另以原 FP32 arrays 重建完整GEMM、quicksort及雙方向any-positive事件。沒有把CORE caption fraction、BF16 GPU dot、stable ties或25000句分母混入。COCO/Flickr各自所有事前主方向R@1須在1pp內，兩個不同canonical benchmark都達標才可資格PASS；同集方向／子集不能充數。

最終硬化已確認：報告綁原index receipt、smoke、source commit、完整settings／實際runtime及原FP32 arrays；checkpoint逐batch核fingerprint、原邊界、每batch SHA及全分母。非產生者重算全部雙方向指標，明確接受事前揭露的port／Flickr mapping差異後，才可產獨立收據。重算者自己的NumPy實際path／module version／distribution version也必須與釘版依賴相同，並寫入收據；竄改此欄、原分數／array／state／model／batch／source或缺benchmark皆拒絕。

官方HF PyTorch與原JAX、PIL與TF resize、Karpathy與TFDS列序、Flickr原loader未公開等差異，均在GPU前揭露，不能宣稱exact JAX重現，也不能看效果再挑前處理。這些差異的最終資格接受與獨立重算尚未執行，不由技術PASS預先代替。

中央入口只做smoke及兩個完整index，不自動分析或讀分。固定20,480MiB保守門檻，維持中央global2／每卡1及既有程序數限制。P2/P3仍須真P0完成與非產生者資格PASS；結果讀取用另凍結的analysis PREREG登記actual time與來源plan/commit，保留仍凍結producer的原始bytes。
