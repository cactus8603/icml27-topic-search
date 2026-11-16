# [codex] CORE2 P2 候選：獨立計畫與資格證據審查

[codex] 2026-11-17 00:21（台北）  
Reviewer：`/root/independent_numeric_review`

**PASS：可只將 state 改為 FROZEN、reviewed 改為 true，另建新的 approved plan 並 commit 凍結；其餘內容須完全不變。** 本報告沒有建立 approved plan、修改候選／來源、commit 或使用 GPU。這是既有 P0 資格證據與新 P2 計畫的審查，沒有讀取新 P2 output／效果，不代替使用者的科學定案。

## 審查範圍與先登記讀取

- P2 source：`icml27-topic-search@f69b876213f98d1aa53788a62ecb6c67a9f233e0`。七份執行來源／PREREG 均與該 commit 原 bytes 相同。
- 候選：`codex/retrieval_matrix_panel_runtime_next/p2-core2.review.v1.json`，SHA256 `33ca7c2b88da48027335e1973cdf83d302b1cab09671273199453187a7dfcd5a`；實際仍為 `REVIEW_REQUIRED`／`false`。
- 分析／讀取協定：`icml27-topic-search@7b4855f5d1b7f6e6c1f3850b8a2a9e41b5ac82ed`，approved analysis plan SHA256 `c201108db0f6cc5d923f07aa16f2d26b77588656ce38507821aebd3fb42576fe`。
- 在打開候選或真 Q 前，先通過原 frozen analysis source gate、實際 NumPy runtime 與 append-only journal；再以 fixed-embedding-core2 job 登記完整 producer read_paths、mechanical／independent／qualification receipts、候選及本報告 prospective path。
- 本次 READ_STARTED hash：`2c446510a5dec6983b51e7a327b112abbc1fe56e9033dc48e9ebbcd5cdab50d9`。終止紀錄由同一 ReadSession 寫入各實際檔案 hash；analysis immutable body 與 producer PREREG 未改。
- 我沒有實作模型、評估、分析或 P2 接線，只核查已獲明確授權的 CORE2 P0 evidence；不讀其他 job 或新 P2 output。

## 真資格、原來源與模型不變

- 真正原 `Q.admit` 與新 `C.admit` 均通過，且兩者回傳資格完全相同：state PASS、matched distinct datasets 3/3、原 parent plan 與完整 model spec 均精確綁定。
- P0 producer：`icml27-topic-search@99431594893bba94337b3befd10a7a483ba7374f`；plan `/ssd4/cactus8603/icml/codex/retrieval_matrix_core_runtime_fix/p0-qwen3-fixed-v1.json`，SHA256 `0f74cb526e08952e05cc178e9fc2fba91d773701db7d92049a71661a142049b1`。
- 三份 P0 index 均 COMPLETE，identity schema／owner／model／dataset／plan SHA／994 full ref／完整 spec／settings／17-entry 原 source 表全部相同於固定原來源；表內含原 plan 與 PREREG。
- 三份 processor.json payload SHA、固定 implementation／loader／processor configuration／batches、CORE compatibility marker，以及全部 recorded runtime 完全相同於候選的 qualified parent signature／runtime。
- Checkpoint：`Alibaba-NLP/core-emb-2b@cc9870ea92d64ebd5f8f9fc368d1120453564158`；完整 spec SHA256 `e5ec104bc9d480a60dbd2536b2db13ffc75b4ec15bcc66f8e1896452da92f416`。
- 原 loader：`{'torch_dtype': 'bfloat16'}`；原 batches：`{'cross': 4, 'image': 4, 'text': 32}`；原完整 settings SHA256 `5bbd678b4077ddc4a0348e8e4dabe0dbf269f25c4839e607be9b76d65a12b8ca`，dependencies SHA256 `f5ea3ec67d5d63eb10de88bf6e9cc32e0f0c13bebcd9db347f125afb7ab9451a`。
- 候選保留原完整 spec、model／processor／batch／dtype／settings／dependencies，不增加調參或新模型。builder 以真 Q 與 audit 在記憶體重新產生的候選，與這份 JSON 完全相等。
- Recorded runtime：torch `2.14.0+cu126`、transformers `4.57.3`、CUDA `12.6`、device `NVIDIA GeForce RTX 3090`；修復 marker 與固定 spec 相同。
- 原 P0 沒有實際記錄 seed／TF32／matmul profile；候選保留同一宣告與 disclosure，沒有聲稱驗證未知狀態或補造原收據。

下表只是既有 P0 資格的固定對照，不是新的 P2 成績：

| P0 dataset | 官方欄位 | 已資格 observed | 原 target | 差值 pp | 原 ±1 pp |
|---|---|---:|---:|---:|---|
| `coco` | `coco_t2i_recall_5` | 0.792556169 | 0.793000000 | -0.044383 | True |
| `flickr` | `flickr_t2i_recall_5` | 0.961200000 | 0.962000000 | -0.080000 | True |
| `negbench` | `negbench_coco_recall_5` | 0.765811146 | 0.766000000 | -0.018885 | True |

- 原 mechanical COMPLETE 與正式非實作者 independent PASS 都綁相同分析 plan／commit；兩者成功讀取事件仍在 durable journal，正式獨立 receipt 亦綁 exact mechanical 與 reports。
- Mechanical SHA256：`f4316e486627e2570c324e7602ae19278553e564a9bdbe13849f4ecd54177338`。
- Independent review SHA256：`c5b312f3ea2332a7bcae146df8f6b84bfc7ade6f1ce59afbccb72077bca33693`。
- Qualification SHA256：`71545c9b3b0d8005fa40b9513782d37a0c3132119ce8555eacb9899da4f7aa85`。

## Dev-v2、COLA 與評估設定

- 真 development audit 通過：state `PASS`、independent recomputation `True`；audit／manifest／development source／images summary／images receipt 的 SHA 與候選 entries 及彼此關係全部一致。
- 通過既有 `data_audit.verify`、`bind_datasets`、exact `verify_datasets`；完整四 datasets 與固定 adapter 重新生成一致。這是核查既有完整獨立 audit 的綁定，未在此重新 inference 或重新建立 development data。

| Dataset | Split | Gallery | Queries | 固定用途 |
|---|---|---:|---:|---|
| `cola-gqa` | public | 2,589 | 320 | fixed baseline test only; no fitting or method selection |
| `cola-paco` | public | 7,921 | 400 | fixed baseline test only; no fitting or method selection |
| `conjunction-dev` | dev | 20,000 | 1,089 | formal conjunction development |
| `negation-dev` | dev | 20,000 | 237 | negation diagnostic only; insufficient 1000-query power |

- 正式 conjunction 至少 1000 queries、全庫 20000；negation 只作診斷。COLA 只作固定 baseline test，原 caption adaptation／threshold-group AP 與 GQA 2589／320、PACO 7921／400 均保留，不作方法選擇。
- 全庫 FP32 CPU dot、gallery-order stable ties，K100 主，K50／200 補充；ignore 僅於評分移除，不補 K。這些 panel 排名不冒充原 P0 native BF16 paper table。
- P3、新方法、COLA 方法選擇及新的 native paper ranking reproduction 仍明確不在候選範圍。

## 允許的後續動作與界限

僅允許另存一份內容相同、只改 state／reviewed 的 approved plan，之後連同來源與 PREREG 凍結並確認執行目錄乾淨；本報告沒有代做這些動作，也不是 GPU 提交。後續只能透過中樞唯一中央排程，沿固定四參數 launcher 與單卡 assignment 執行。
正式 P2 效果仍須另外凍結讀取／分析協定，再依 mechanical → 非實作者原 arrays 獨立重算 → 使用者定案。root／matrix_runner 及相應 backend、runtime-fix、analysis 實作者不能作該正式數值 reviewer。
