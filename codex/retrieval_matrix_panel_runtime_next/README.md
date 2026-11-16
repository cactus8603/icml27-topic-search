# [codex] Repaired MLLM fixed P2 bridge

獨立source接線，支援明列9943159修復後CORE／Qwen3及Qwen2模型。真正三benchmark P0資格先admit；完整parent、spec、settings、processor、runtime與依賴一致才讀dev-v2／COLA，無真Q無P2plan。候選需另review／freeze。此處沒有正式plan、GPU執行或模型效果。

builder：`python -B codex/retrieval_matrix_panel_runtime_next/build_plan.py --family FAMILY --model MODEL --qualification REAL_RECEIPT --development-audit REAL_AUDIT --out NEW_CANDIDATE`。兩個family只接受contracts.FAMILIES的固定parent。run.sh沿中央四參數FULL_REF PLAN MODEL OUT，選原隔離Python與依賴，只smoke／index；不得子代理自行執行。

原模型與修復來源不修改，原全局patch runner不import；新入口要求新process。原P0未記錄seed／TF32實際profile的界限保留披露。P3、正式分析及研究判斷另處理。
