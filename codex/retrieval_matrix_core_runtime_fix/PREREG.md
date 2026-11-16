# [codex] CORE/Qwen3 technical API compatibility repair

2026-11-16。原embedding與MCMR合成smoke失敗：作者wrapper用bare `@check_model_inputs`，釘版Transformers4.57.3改為decorator factory，forward被誤換成只接func的wrapped_fn，收到input_ids即TypeError。沒有執行任何本批正式資料推論，沒有讀主指標。失敗收據／log／原plan與原source保留，不覆寫或自動重試。

此版不改任何原作者或已凍結來源／依賴檔案。只在新的程序中，從這個已知bare-decorator closure取出未修改的原forward，再以釘版library的 `check_model_inputs()(original_forward)` 正確套用。嚴格檢查library版本、原module/class/function/closure形狀；不相符即拒絕，不提供其他fallback。不能略過decorator、改attention、pooling、normalize、processor、dtype、batch、template或checkpoint。

新plan保留原model spec全部欄位，僅新增明確runtime_compatibility；parent plan逐SHA綁定，原settings/datasets/dependencies須相同。CORE2/8、Qwen3VL2/8共用同一已釘版wrapper；MCMR原生renderer／完整商品／native BF16計分不變。程式與新plan必須經非實作者審查並commit，再由root唯一中央派工，輸出用全新目錄。模型效果仍須完整P0＋獨立重算；技術修復不是資格PASS。

CPU regression使用真作者model class的小型隨機權重，實際走文字與多模態forward，先重現原TypeError，再比較修復後與原未包裝forward body的每個hidden/attention tensor逐bit相同。CPU只用合成tokens/pixels，不開benchmark或初始化CUDA；正式GPU仍有原固定batch invariance smoke。修復若再失敗保留紀錄，不以降低batch、改精度或同分規則解套。

GPU由中央賦予owner/task及單一卡位。分析／資格檢查另行執行，讀主結果前在獨立analysis PREREG登actualread time與原plan/ref，保持producer凍結bytes。
