# [codex] CLIC paired native P0：CPU 準備交付

本次是實作者的準備收據，不是獨立資格審查。三個 CLIC 固定公開 checkpoint 已有完整來源、可執行 wrapper、兩個完整 benchmark 與 CPU 驗證；P0 均為 NOT_RUN，尚未凍結、排程或使用 GPU，沒有讀取正式模型效果。

實作位於 `codex/retrieval_matrix_paired_p0/`，沒有修改 frozen extra/native/main/qwen2/rerank/dispatch。只重用原固定 checkpoint／依賴／作者 source；SugarCrepe 七個官方 JSON 額外下載並釘 Git revision/blob/SHA。全部圖檔從既有 COCO cache 取得。

## 完整資料與模型身分

SugarCrepe 為完整 7,511 題、七子集；SugarCrepe++ 為完整 4,757 題、五子集；共同需 1,560 個獨立圖檔，沒有缺圖或排除題目。各題均保留原 image row 與 raw caption，不去重、不套 retrieval pre_caption30、不清除相同正負句。SC++ 採其作者改過的原始正負句與 caption2，不在本機拼接 SugarCrepe。

固定 CLIC-LAION/CogVLM B32、OpenAI L14、CLIPS L14；published targets 分別對應 CLIC paper v2 Table1、Table3 第一段與最後一段 CLIC-LAION，沒有使用 RedCaps 列。SugarCrepe 與 SugarCrepe++ 是兩個 canonical benchmark；子集或 ITT/TOT 不另外計資格數。每個已宣告主要方向及 group macro target 都須距原文不超過 1 個百分點。

重要更正：作者 factory 只有大寫 `CLIPS` alias 才選 `CLIPS_Tokenizer`。舊 frozen extra 的小寫通用 alias 會走另一 tokenizer；其技術預檢不能作原生資格。本模組使用作者 CustomTokenizer（BOS1/EOS2/pad0/末尾 class101/長度80）與 vocab.txt，模型本體依作者 modelcard 走 OpenCLIP2.31.0。新 spec 明列此 task-specific 政策，保留 unchanged extra acquisition spec digest，沒有改舊 frozen bytes。

## 可執行與計分

每張圖及原句均 batch1、完整 FP32，實際呼叫 AST 取得的未改作者 load_encode_img/load_encode_text；原 F.normalize 保留。SugarCrepe 為逐元素 dot、strict greater-than；SC++ 為作者 CosineSimilarity(dim1,eps1e-6)，ITT 與 TOT 各有兩個 strict 條件。Equality 全錯；各子集完整 accuracy 後再算未加權 macro，不能 pooled。

GPU producer 日後保存原 FP32 embedding arrays、每個 batch/state SHA、逐題六欄 scalar arrays 與其原 checkpoint。CPU analyze/review 重驗 original embedding receipt、完整 batch provenance、同一 runtime/smoke/settings/source freeze；不同 reviewer 另從原 vectors 重算，要求每題 strict event 均與 GPU scalar 一致。任何 CPU/GPU tie event 不一致即拒絕。NumPy/Torch actual path/version/distribution 與 facade source 都綁入 review，admit 重驗，不以技術 forward PASS 代替研究資格。

接口為 `codex.retrieval_matrix_paired_p0.qualification.admit(entry, model_spec)`；沿用 schema2 receipt，但必須走此 family-specific facade。不能用 SigLIP2 Q.admit 計分後端直接驗 CLIC。新 paired model_spec 的 batch1/raw/tokenizer 即為資格身分，P2 如改 batch 或文字政策不能沿用該資格。

`central_task.sh FULL_REF PLAN MODEL OUT` 已備妥供 root review，依序 smoke、SugarCrepe index、SC++ index，沒有自動 analyze 或派工。run.sh 的 audit/analyze 是 CPU 階段；所有正式 GPU 入口仍要求 root central 環境、clean Git source/plan freeze。

## CPU 驗證

9 個新 paired 測試加 14 個 frozen native 測試，共 23 tests PASS，5.698 秒。新測試包含 raw caption/labels 隔離、完整分母、strict ties、macro/pooled 差異、正確 CLIPS tokenizer＋OpenCLIP meta model、無 native module global mutation，以及 synthetic 原 arrays→checkpoint→獨立 review→目標 FAILED 保留的完整證據鏈。另拒絕改方向／分母／aggregation／subsets 或移除 source file。synthetic 例刻意全部 strict ties，沒有造 P0 PASS。

精確合併命令：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TMPDIR=/ssd4/cactus8603/icml/hf_home/tmp \
PYTHONPATH=dl/retrieval-matrix-extra/deps:dl/retrieval-matrix/deps/qwen2-transformers451:dl/retrieval-matrix/deps/qwen2-flash:dl/retrieval-matrix/deps/core-p0 \
dl/retrieval3090/envs/retrieval3090/bin/python -B -m unittest \
codex.retrieval_matrix_paired_p0.test_paired codex.retrieval_matrix_native_p0.test_native
```

三模型最新完整 CPU audit 均 TECHNICAL_PASS：全部 2,766 dependency payload、原資料／圖檔 SHA、actual module paths/profile、所有 checkpoint keys/shapes、actual tokenizer 與 RGB transform 均核，CUDA initialized 均 false。B32 為 302/302 keys；L14 446/446；CLIPS L14 444/444，沒有 missing/unexpected/shape mismatch。兩個 shell 入口 `bash -n` 通過。

三模型另以完整真權重及作者 helper 跑 CPU 純色 JPEG／一原句／空句 forward，合計 24.870 秒。image 1×512/768/768、texts 2×512/768/768，均有限 FP32 unit vectors、所有參數 CPU FP32、CUDA 未初始化。其後只補 qualification/test 的精確 descriptor/source 拒絕閘門；未重跑相同模型 forward，已保存三份舊 plan payload，並逐一證明新 plan 的 model_spec、models.py、checkpoint/tokenizer/helper bytes 完全相同。

## 最新凍結候選與收據

| 檔案 | SHA256 |
|---|---|
| p0-clic_b32.json | 0e47106fba308ca6ef98e4156b46f832c96d201482685dc9f7b79a4f6b130d8d |
| p0-clic_l14.json | 70ef33780e610f51776a66d29a806a1b4ff35708604777f590cc447e03e76f88 |
| p0-clic_clips_l14.json | e333ed2e3966b14369ce66fc02c0c6fc66c36a1baac87b42a0392ff58df95918 |
| preflight-final-v2/clic_b32/cpu-audit.json | 8a80ed4f95acdc6b7f02557bc9ea8965843bbfa64be3406e6992956480eb49c9 |
| preflight-final-v2/clic_l14/cpu-audit.json | 0b04815746b6821c42028d23408f4581a80eb3c26b0245d4cf66a1800e3e4d1c |
| preflight-final-v2/clic_clips_l14/cpu-audit.json | 205b6b063477032d2495657f6b5085908baf4e31cfddb1834cac66080e26e0e9 |
| cpu-synthetic-forward-final.json | 32208f85dcff93c92376803b172fdddddcbdd9cca5c3ff541799b45d4cbf8079 |
| cpu-forward-plan-links-final.json | db20b24a24c9cce9d97ab44e494cff616e6d9057fbc223dbe7b164850e9c56ad |
| data-audit.json | 2919f7f72787e9dce3a51e5c6341dfcc3f228763ef7d751d1a7f7119f994995e |

前三個 plan 在新 codex 模組，其餘收據根目錄為 `dl/retrieval-matrix-paired-p0/`。舊 preflight 與 forward 所綁舊 plan 留在 provenance/pre-gate-update，不覆寫技術收據。

界限：固定環境 Torch2.6cu124 與作者 Torch2.2.2cu121 不同，TF32 本機明確禁用；cudnn.benchmark=True 保留。按 modality 快取 batch1 vectors 後重放原 scalar，而非每題交錯 encoder；原 training seed 與 paper 表格 run 的映射未完整公開。這些差異已進每模型 plan 的 disclosed scope，必須由非產生者獨立接受並完成真 GPU native P0 才能 admission。目前沒有資格 PASS，也沒有送出任何新 GPU 工作。
