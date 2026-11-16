# [codex] CLIC paired P0 獨立啟動前審查

審查者：Codex root（非實作者）。時間：2026-11-16 22:50，台北。結論：三個固定 CLIC checkpoint 的 CPU／source／protocol 檢查可進入凍結後中央 GPU smoke。尚無實測 P0 資格，沒有讀任何正式效果。

逐讀新 model、data、metrics、runner、qualification、builder、shell 與 PREREG；獨立核 cached CLIC paper v2 Table1、Table3 第一與最後 CLIC-LAION 區塊，七個主要 grouped targets 與 variant 對應一致。核未改的作者 general_eval.py 與 eval_utils.py，batch1、raw captions、FP32 F.normalize、SugarCrepe elementwise dot 與 SC++ CosineSimilarity eps1e-6、ITT/TOT strict comparisons 均相符。CLIPS 正確使用含大寫CLIPS alias的作者 CustomTokenizer，沒有沿用 frozen extra 小寫 alias 的錯誤 tokenizer；新 spec 明列差異且綁原 acquisition digest。

root 重跑9個 paired＋14個 frozen native tests，共23項PASS，5.258秒。另獨立執行三份完整 audit，各2766依賴 payload、原題／影像 SHA、actual imports、tokenizer、meta class keys/shapes 全 PASS，CUDA false。B32 302 keys、L14 446、CLIPS 444均完整匹配。data保留 SugarCrepe7511、SC++4757題、1560獨立圖，paired image rows不去重；raw captions未套 pre_caption30。實作者真權重 CPU 合成 forward 與 pre/post plan等同性收據已核，但 root 沒有無故重跑完整 CPU forward，也未 GPU。

獨立 qualification facade 綁完整 source／model／settings／runtime／smoke／array batch receipts，要求原 GPU scalar 的每題 strict events 與 CPU FP32重算一致，再由不同 producer reviewer 重算所有 grouped metrics；保留兩個完整 benchmark，不能以方向／子集充作兩個資格。synthetic receipt鏈測試刻意全tie且維持FAILED。正式admit必須走此 paired facade；後續P2須明確接線且同一paired model_spec，不能直接套 SigLIP2 backend。

啟動前接受已明示的本機Torch2.6cu124（作者2.2.2cu121）、TF32禁用、cudnn.benchmark=True、按modality快取batch1後重放相同scalar，以及作者未完整公布table-run seed映射等限制；不將其稱為kernel逐bit exact。正式資格仍須真正GPU結果與非產生者逐題重算，任何strict event差異即拒絕，不放寬tie或改精度。

plan與root獨立audit：

- p0-clic_b32.json: `0e47106fba308ca6ef98e4156b46f832c96d201482685dc9f7b79a4f6b130d8d`；root audit SHA `8a80ed4f95acdc6b7f02557bc9ea8965843bbfa64be3406e6992956480eb49c9`。
- p0-clic_l14.json: `70ef33780e610f51776a66d29a806a1b4ff35708604777f590cc447e03e76f88`；root audit SHA `0b04815746b6821c42028d23408f4581a80eb3c26b0245d4cf66a1800e3e4d1c`。
- p0-clic_clips_l14.json: `e333ed2e3966b14369ce66fc02c0c6fc66c36a1baac87b42a0392ff58df95918`；root audit SHA `205b6b063477032d2495657f6b5085908baf4e31cfddb1834cac66080e26e0e9`。

審查來源SHA256：

- PREREG.md: `8727000037c7d58d80617ad763c68d55ccc326010893f881798b4c9b7ee25e91`。
- SOURCES.md: `0a66395d18b35af6934ad01c5243668f4bc6434f9450a9a94091176ca91a6240`。
- acquire.py: `3f83a41a232803470978999c8401dfac67a7f12dd46117d8400ad9083f5bf853`。
- build_plan.py: `03f548c754db53f6a6deca1d5ae44d4759cd4b9025c79a1761af7a90054f85bf`。
- central_task.sh: `17a7e8975f9fca3908320b5c545f749d99fd94569edbf5c588cfe6f485e7b761`。
- data.py: `6362d58722bd15207793c7fe3e79740affe5fcdd304d028217c16215c9cd852e`。
- metrics.py: `d7ba0272730eb0f3969a089e857886511b8fc2dc7fcd06df1c92ee104dfa4ef3`。
- models.py: `200e965ca54f2d2479bc16a464f98e8ac73a82859a99fa7ddf0590ac9a6b0699`。
- qualification.py: `09d859fa2f00a887bd152917a6a231ace178a360a50150eee594a7c3d0adb1d8`。
- run.sh: `3ebb5904e6ce9c1d62be2808f7953adeb1ea2eec9a84cba6af99c6572d8f775f`。
- runner.py: `4630e91943b94b9bfc61892fe736d5e53b4e7ea92b66f38f34903f118a319ff3`。
- test_paired.py: `7ff464bde051df1ca3eb7ae9ef79db0c6c3d2e6f54797f313915b222fbe6f376`。
