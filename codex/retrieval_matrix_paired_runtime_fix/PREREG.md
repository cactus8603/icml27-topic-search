# [codex] 三個 paired CLIC 的獨立 runtime 技術修復

[codex] 2026-11-16（台北）。本批只準備CPU／source，不GPU、不排程、不重試、不讀模型效果。三個parent plan固定54f544c；model_spec、原作者PairedModel class／encode、raw tokenizer、完整SugarCrepe7,511／SugarCrepe++4,757、strict事件／macro、FP32 batch1、seed1、TF32禁止與cudnn.benchmark=True完全保留，不用Google scorer替代paired公式。

技術修復一：原runtime在CUDA可見時，peft→Bloom→flash padding會需要未列出的einops。CUDA hidden的lazyimports PASS不能證明完整；新CPUprobe必直接import flash_attn.bert_padding並實際unpad/pad，核原2766檔／9modules加已凍結9943159 overlay32檔／einops0.8.2，完整actualpath/version/distribution精確核對。原依賴與任何frozen source不改。

技術修復二：原signature transform repr包含 '<function NAME at 0x...>' 的記憶體位址，跨process的smoke/index會誤拒。新factory仍返回原PairedModel exact class，只在新instance的signature收據刪除此volatile地址，保留函式名稱、transform class／全部參數及其餘內容。原transform／encode／模型／tokenizer不改；來源／payload SHA仍固定。CPU跨process構造原transform/tokenizer驗穩定性，不重新載入真模型。

新qualification以private module重用paired專用Q及其完整原GPU scalar／vectors／checkpoint／事件獨立重算；只在canonical科學規格驗證時投影回原dependency/source欄位，真證據始終綁新plan與10-module runtime。非產生者須另接受原已披露協定差異。新scope official-paired-runtime-fix-p0-qualified拒舊P0／Google資格；沒有真實數值PASS，也不能借技術CPU收據當資格。

三個新plan初始REVIEW_REQUIRED／reviewed=false。root獨立review、完整source／plan commit、clean、中央owner/task及單GPU後才可進新out；原所有輸出保留，不fallback或減batch。launchers四參數FULL_REF PLAN MODEL OUT。已有真modelCPUhelpers收據保留，以完整model spec、原模型source／payload/tokenizer/helper bytes等同性鏈承接；不無意義重跑真權重CPUforward。

P2必重用這個same class／signature policy／runtime與專屬qualification，不自動接舊P0gate。正式效果另凍結analysis PREREG、登記本producer plan SHA／commit及實際讀取，producer bytes不改。
