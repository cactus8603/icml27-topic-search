# [codex] MLLM P2 原 producer identity 綁定修復

2026-11-17 00:06（台北）。尚未凍結的新 P2 接線由 matrix_runner 實作，獨立 reviewer 發現原 Q.make/admit 只核 plan/model/settings，未核 report identity 的來源 ref/source 表。非實作者以完整暫存合成 reports/review 經真正 Q.make→Q.admit，確認 wrong ref/source 可通過初版新入口；沒有讀真模型結果。

因當時四個 active slots 已滿，root 接手新 contracts/tests/PREREG 的最小修補，成為本批接線共同實作者。固定原 producer source_gate 的完整表由 base.source_files 加原 plan 本身及原 PREREG 構成；每份已資格報告須 schema/owner/model/dataset/plan SHA/full ref 994/full spec/settings/source 表精確相同、index COMPLETE。額外核實際 environment 的原 spec 版本與 processor implementation/loader/processor configuration/batches，不能讓三個同樣錯誤的 signature 互相通過。新 P2 source 表不冒充原 P0 producer 表。

新增真正 Q.make/admit 的兩 family 合成正向及負向測試，確認舊 Q 可接受但新入口拒 wrong ref、wrong/漏/多/改來源、單一 benchmark 錯來源、非 COMPLETE、錯 environment 及錯 static signature。全部資料僅 hf_home/tmp、測後清理，未讀真 Q 或模型效果，未 GPU/改原 live/frozen 來源。

root 執行 19 新加 13 原共 32 CPU tests，22.853 秒全部 PASS。此為實作準備而非獨立審查；仍須 fresh reviewer 重跑真 Q 反例與全套後才 source freeze。原實作者交付表為先前版本歷史，不代表此修補已獲 PASS。root 與 matrix_runner 皆排除後續 P2 正式數值審查。
