# [codex] Repaired MLLM P2 runtime next：獨立 prelaunch review

[codex] 2026-11-17 00:10（台北）  
Reviewer：`/root/independent_numeric_review`

結論：**PASS，限下列七檔版本的 source／CPU 準備審查**。目前沒有正式 P2 plan；此結論不是 P0 資格、P2 效果、GPU 啟動授權或使用者定案。真 qualification 採用後產生的候選仍須另行審查、凍結及檢查來源乾淨，才能交中樞排程。

本輪僅讀 source／指定交付與契約稽核報告、原凍結 plan metadata，執行 CPU 合成測試及暫存探針。未讀真 Q、真模型 arrays／reports／效果，未構造真模型、操作 GPU／排程／程序或 commit，未修改實作、原 frozen／live folder，未搬移中樞路徑。我保持非實作者角色。

## 最終審查來源

| `codex/retrieval_matrix_panel_runtime_next/` 檔案 | SHA256 |
|---|---|
| `contracts.py` | `4f51222d616ce8dbf9b8ff56a8d16c72649a39673c6aa2c946c7f32c9f6c51fa` |
| `build_plan.py` | `99adf9c85a3440efe8896ea7866d145a1162240eacd9ae876ed386fe4d4e88fd` |
| `runner.py` | `fecb98a5018768725108fa9eef004ac7cb9cd89567ec105ff74a97480834b72c` |
| `test_contracts.py` | `e5fbfd6775e123f868dbedcde431c6afdb513c0d604dff630a5455579a340b22` |
| `run.sh` | `be411ec5107d66d4f284babce1197031587eb4d5ac1387c3b6c3cc60b99f2ebd` |
| `PREREG.md` | `3adeea1038d9c4bb74dd58eee064feb5ea5a9b45bea4c35438bb5297bca74aa9` |
| `README.md` | `20088acf44cbc946af8ccd439fff2d2118696111852bf78965cf361bf89d8a49` |

固定 producer full ref：`99431594893bba94337b3befd10a7a483ba7374f`。

| Family | 固定原 P0 plan | SHA256 |
|---|---|---|
| CORE／Qwen3：core2、core8、qwen2、qwen8 | `codex/retrieval_matrix_core_runtime_fix/p0-qwen3-fixed-v1.json` | `0f74cb526e08952e05cc178e9fc2fba91d773701db7d92049a71661a142049b1` |
| Qwen2／einops：gme2、vlm2vec2 | `codex/retrieval_matrix_qwen2_runtime_fix/p0-qwen2-einops-v1.json` | `07a9b6859e17d12e99a00439327381655ec22b1369e2d27a9fa4a5f8a5ea3053` |

原 panel／data／audit／qualification 來源固定於 `17e990ba6294b1962f67d0c9a33bccf291e43fbd`。沒有延伸到 CLIP、未修復版本、MCMR、P3 或新方法。

## 實際獨立驗證

使用原 Qwen2 隔離 CPU 環境、原四個依賴路徑、明確空白 `CUDA_VISIBLE_DEVICES`、offline 與 repo 內暫存，執行：

```text
python -B -m unittest -v
  codex.retrieval_matrix_panel_runtime_next.test_contracts
  codex.retrieval_matrix_panel.test_qualification
  codex.retrieval_matrix_panel.test_data_audit
  codex.retrieval_matrix_panel.test_cola

Ran 32 tests in 22.504s
OK
```

共 19 項新測試與 13 項原 qualification／development audit／COLA 測試。另完成全部 `.py` AST parse、`bash -n run.sh`，確認新 folder 無正式 `.json` plan。CPU runtime 測試只替換 private closure 的 CUDA reader 與暫存假 assignment，實際 receipt extension 程式仍執行；沒有初始化 CUDA。

我額外用兩個**各自全新 CPU subprocess** 呼叫真正原 994 修復 producer 的 `source_gate`，不跑 smoke／index，也不構造模型；其回傳完整 source 表與新 `producer_sources` 完全一致：CORE 17 entries、Qwen2 28 entries，兩者 `torch.cuda.is_initialized()` 都是 false。這是原 P0 source 表的核查，不是新入口依靠舊 runner 的全局 patch。

## 發現的阻擋與修復複核

初版 29 項 CPU 測試通過後，我另以 `hf_home/tmp` 內的合成完整報告／review，走**真正原 `Q.make` → `Q.admit`**，沒有 stub 這兩個函式。三份報告雖填入錯誤 producer ref／source 表，初版 `admit_parent` 仍接受；當前磁碟來源正確，不能證明已資格數值出自該來源。因此當時阻擋凍結並通知 root。

root 在尚未凍結的新 folder 內修補後，我逐行核對新綁定，並以自己獨立重建的完整 source 表、原固定 signature 欄位及 spec environment，再跑同一類真正 Q 流程：

| 額外合成探針 | CORE family | Qwen2 family |
|---|---|---|
| 完整有效 baseline | 原 Q 與新入口皆接受 | 原 Q 與新入口皆接受 |
| 只改 producer full ref | 原 Q 接受，新入口拒絕 | 原 Q 接受，新入口拒絕 |
| 只改完整 source 表 | 原 Q 接受，新入口拒絕 | 原 Q 接受，新入口拒絕 |
| 只改 Flickr 一份報告的 ref | 原 Q 接受，新入口拒絕 | 原 Q 接受，新入口拒絕 |

上述六個負向例均在新 producer identity／source／ref 檢查拒絕；有效 baseline 先通過，沒有用別的不完整 metadata 導致失敗來冒充修復。新版 suite 另覆蓋漏／多／改 source、缺原 plan entry、非 COMPLETE、錯 environment 及錯 static signature。**此阻擋已解除。**

## 契約核對結果

- 真正原 `Q.admit` 必須先成功，且回傳資格須綁 exact parent plan entry／SHA、model、完整 spec；之後才可讀 development audit 與 dev-v2／COLA metadata。無資格的合成 stub 在 audit／panel reader 前被拒絕；builder 只產 `REVIEW_REQUIRED`／`reviewed=false` 候選。
- 三份 P0 報告各自須 index COMPLETE，identity 的 schema／owner／model／dataset／plan SHA／994 full ref／完整 spec／settings／原完整 source 表皆一致。原 source 表包含原 plan 與 PREREG，不能由新 P2 source 表替代，也不能只驗目前磁碟 source。
- 各 processor payload SHA 須對原報告；固定 implementation／loader／processor configuration／batches 對原完整 spec；三 benchmark 的 actual signature 與 recorded runtime 須完全相同。CORE signature／runtime 的 compatibility marker，及 Qwen2 的原 overlay manifest／parent ref／actual einops module receipt 皆綁定。
- P2 完整 spec、原 native settings（含 seed 宣告）、batches、BF16 loader、依賴與原 class 保持不變。CORE 使用原 `CompatibleOfficialModel`；Qwen2 使用原 `Qwen2OfficialModel`，VLM2Vec-V2 的 DoRA 載入／merge 仍由同一原 adapter 執行。建構後 actual signature 須等於 qualified parent，才可 encode。
- 新入口建立 private matrix module 與獨立 model namespace，不改 canonical `R.models`／gate／runtime／data globals，不 import 原 panel／Qwen2 runner 來拼接。若共享 model dispatch 已被舊入口 patch，會明確要求 fresh process；不依靠猜測 import order。
- 執行時再核原全部 dependency bytes／distribution、Qwen2 overlay payload／actual modules、指定隔離 Python entry、中央 owner／task 與單 GPU assignment。新 runtime 除中央 GPU UUID 外，所有已記錄的 module paths／versions／CUDA／device family／repair 欄位都須與 qualified P0 相同。
- `run.sh` 使用中央四參數 `FULL_REF PLAN MODEL OUT`，按固定兩 family 選原 Python／依賴路徑，只執行 smoke 與四個固定 index，不自動 analyze。輸出限定新 `dl/retrieval-matrix-panel-runtime-next/runs/`；source gate 要求所有原／新執行來源 Git bytes／SHA、完整 source 表與相關來源目錄 clean。
- 原 P0 並未實際設定或記錄 seed／TF32／matmul profile；新 PREREG 明示此限制，保留原已記錄語義，不補造舊收據、不悄加新 profile、不聲稱驗證了未知狀態。新 profile 若必要須另做 P0。
- 開 dev-v2／COLA 前須已真正 admit parent；執行時再驗 development audit／manifest／source／images summary／receipt 與 exact dataset entries。正式 conjunction 至少 1000 queries、全庫 20000；negation 只作診斷；COLA GQA 2589／320、PACO 7921／400 固定 baseline test，不作方法選擇。
- 排名維持 FP32 CPU dot、gallery-order stable ties；K100 主、50／200 補充；先在全庫取 K，ignore 僅於評分移除、不補 K。COLA caption adaptation 與 threshold-group AP 的差異保持披露，不冒充原 native paper ranking。
- 正式 P2 效果仍須另凍結 analysis PREREG 與讀取記錄，再依 mechanical → 非實作者原 arrays 獨立重算 → 使用者定案。接線作者 root／matrix_runner，以及相關共用 backend、runtime-fix、分析實作者均不得擔任相應正式數值 reviewer。

本輪結束時未產生正式 P2 plan，沒有讀取新 P2 效果；既有 `codex/retrieval*`、原 analysis、live central／NEXT／v3／base 均保持原路徑與內容。
