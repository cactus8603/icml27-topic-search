# [codex] CORE runtime API repair：獨立 CPU prelaunch review

審查者：Codex matrix_runner（非實作者）。時間：2026-11-16 22:31，台北。

結論：在本次 P0 embedding 與原生 MCMR 修復範圍，未見阻擋啟動的程式問題。修復只對固定 Transformers4.57.3 的 decorator factory API 進行兼容，沒有更改原作者 source payload、checkpoint、processor、template、pooling、normalization、attention、batch 或精度。技術預檢通過不代表正式 P0 資格或研究效果通過。

原始作者 wrapper 的 bare `@check_model_inputs` 在此版 library 被當成 factory 參數，closure 中唯一 nonlocal 為指向原 forward 的 tie_last_hidden_states。compat.py 檢查固定 module/class/function/closure 形狀，取回同一原 forward，再以真 library `check_model_inputs()(original)` 正確套用。重複修復只允許相同 marker/reference；對不相符形狀拒絕，沒有另外的計算 fallback。

獨立重跑三個 CPU tests，全數 PASS，6.564 秒。命令使用 `dl/retrieval3090/envs/retrieval3090-torch214/bin/python -B -m unittest codex.retrieval_matrix_core_runtime_fix.test_compat codex.retrieval_matrix_core_runtime_fix.test_plan`；PYTHONPATH 僅優先 `dl/retrieval-matrix/deps/core-transformers457` 與 `dl/retrieval-matrix/deps/core-p0`，CUDA_VISIBLE_DEVICES 為空。真作者 model class 使用 tiny 隨機 CPU 權重與純合成 tokens/pixels：先重現原 TypeError，修復後 text/image 的 last_hidden_state 與 attention_mask 均逐bit等於原 forward body。plan tests亦拒絕 model batch、processor max_pixels、seed、dependencies 或 datasets 變動，並確認 import 無安裝副作用與無 central owner 時拒絕 GPU入口。

另作四路獨立 tiny CPU probe：正式呼叫通常省略 use_cache，因此分別在 FP32／BF16 × text／image、use_cache 完全不傳的預設路徑，比較修復後 model call 與同一未包裝原 body。四路 last_hidden_state／attention_mask 均以 rtol=0、atol=0 通過，且 torch.cuda.is_initialized() 為 False。此 probe 沒有下載或載入真正模型權重，也沒有讀 benchmark texts／labels 或主結果。

已讀呼叫路由：embedding install 將同一 M.OfficialModel 換為 CompatibleOfficialModel；MCMR NativeOfficialModel 在實例化時才讀 M.OfficialModel，因此可使用修復 class。panel 的 BaseModel 於 repair install 後 import 取得。原 P0 與 MCMR 的 parent model/settings/datasets/dependencies 由逐SHA parent plan及 validate_repair 保持相同；P2 的不同資料設定仍須另經原 panel qualification gate，本審查不替代 P2 資格判讀。

本次只讀審查與 CPU synthetic 檢查，未修改實作者檔案、未 GPU、未送排程、未取消或觸碰現行 MCQ worker、未讀模型效果。GPU完整權重與實際 frozen batch 的 smoke 仍由唯一中央排程於 commit 後執行，失敗應保存並停止而非自動重試。

審查版本：

- `codex/retrieval_matrix_core_runtime_fix/compat.py` SHA256 `0097af5627801bc52691870293e74b578958348f6a87586af53307239c96b645`
- `codex/retrieval_matrix_core_runtime_fix/runner.py` SHA256 `91f6cc66afca8ebf75a95999f70b1e5b0b52db94193eb17441436ae992f1a693`
- `codex/retrieval_matrix_core_runtime_fix/build_plan.py` SHA256 `bfa81f37002422d72df5578f6b754282a8da476f8af18127bf9c9512dfb0f999`
- `codex/retrieval_matrix_core_runtime_fix/test_compat.py` SHA256 `8d5b5fd27d7c901650915b56b04e8398684e98011792be53d1c0a91e428876e0`
- `codex/retrieval_matrix_core_runtime_fix/test_plan.py` SHA256 `f7842baea5ea805b699ed3fad870f1eb1676125fc62ff8cc3dc1462a8f788929`
- `codex/retrieval_matrix_core_runtime_fix/run.sh` SHA256 `3d947db7d9c5a1cc25fe4c1c985833ab542715a4b8b1bf4c098e1dd2256f4f8c`
- `codex/retrieval_matrix_core_runtime_fix/PREREG.md` SHA256 `adc2414c161aaa9c228149d2908e74aefb2c17875dd82bf6b19c544239aed8c9`
