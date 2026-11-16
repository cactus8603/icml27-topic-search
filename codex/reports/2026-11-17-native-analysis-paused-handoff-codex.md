# [codex] Native analysis 暫停交接

[codex] 2026-11-17，依使用者暫停指示即刻停筆。此為實作者交接，不是獨立 prelaunch PASS。

本輪沒有讀真實模型 scores／arrays／results／completion／qualification；沒有 GPU、排程、approved plan 或 commit。原 producer 與原 7b analysis 未修改。

已完成：七個 eligible jobs 的完整 prospective manifests、原 mechanical CPU launcher、原生 review→make→admit 橋接、三家族原始陣列獨立重算、MCQ逐題 strict detail audit、固定 CPU modules／NumPy array binary、私有 7b reader／外部 journal、角色排除與拒絕測試。所有程式在 codex/retrieval_matrix_native_analysis/，README/PREREG 已寫正式流程，source 現停筆。

待完成：核收已在跑的最終36-test；非作者 prelaunch review；另建 approved 並凍結；之後才可授權真 mechanical → fresh independent → native qualification → user decision。這些均未執行。

Draft：`/ssd4/cactus8603/icml/codex/retrieval_matrix_native_analysis/plan.native-analysis.draft.v1.json`；狀態 `REVIEW_REQUIRED`、reviewed=False；SHA `6bf340b40b26a1525bd4cbdd14fddd766b332b405b34b9637b92ad45387e2fe3`。

唯一仍在跑的本代理 session：`79247`，腳本 `hf_home/tmp/native_analysis_full_preflight.py`。主機 PID 另由訊息回報；目前 CPU log 已有 30 項結束為 ok，尚未產全套最終結論。不能把局部通過當36-test PASS。

既有測試結論：原 facade17 tests PASS（288.393s）；前一輪 contracts/workflow/pipeline17 tests PASS（138.335s）；更早 source/workflow16 tests PASS（46.155s）；三家族真正 CPU modules/binary positive＋forged-binary rejection 均 PASS。早期 pipeline mock transport 曾誤攔 Git 子程序，已修為只替代 mechanical transport，後續整合通過；不涉及真模型。

其他 sessions 均已結束：66357(exit0，16 tests)、11250(exit0，CPU import)、26724(exit1，早期 mock harness)、63924(exit0，三家族 pipeline)、87239(exit0，CPU binary probes)、37848(exit0，17 tests)、62735(exit0，draft build)；fixture 子代理46450(exit0，17 tests)。沒有其他背景下載或 GPU worker 由本代理新啟動。

最後全套的持久位置：

- `dl/retrieval-matrix-native-analysis/cpu-full-preflight.v1.log`（仍追加）
- `dl/retrieval-matrix-native-analysis/cpu-full-preflight.v1.json`（完成時由既有程序寫出；現在不得預稱 PASS）
- `dl/retrieval-matrix-native-analysis/cpu-environment-{siglip2,paired,reranker}-v2.json`（三家族 CPU runtime 與 binary 收據）
- `dl/retrieval-matrix-native-analysis/manifests/`（七份完整 prospective metadata，非 model effects）

所有 job 的保守 excluded_reviewers 為：`/root`、`/root/matrix_runner`、`/root/public_data`、`/root/official_p0`、`/root/official_p0/p3_contract_review`、`/root/official_p0/native_full_fixtures`；另排除 actual mechanical actor／central producer。known_implementation_authors 僅列有來源的作者，與上述保守角色 union 分開；新 analysis 作者 official_p0／native_full_fixtures 均不可做正式獨立 numeric review。

真缺口維持：qwen_native2/8 與 Qwen8 Instruct 沒兩個精確官方 targets，不 eligible；native-reranker scope 與舊 P3 scope 不相容；retrieval frozen producer 沒完整 GPU calls/peak，不造零成本；native failed/unfinished forward budget 僅上界，hard-kill 未記錄時間／峰值未知。

來源 hash manifest（PREREG僅 immutable body，其他為完整 bytes）：

| 檔案 | SHA256 |
|---|---|
| `codex/retrieval_matrix_native_analysis/PREREG.md` | `4d547cc858d651631b5a8db0aba42e7b4b81bfcdd05e7d9266e0b95bfb406f31` |
| `codex/retrieval_matrix_native_analysis/README.md` | `e215e626892efe65065770885c5b2e1a84ae5415b0a4b2c354056667241c8561` |
| `codex/retrieval_matrix_native_analysis/build_plan.py` | `3391bb7755e255acd9d5836d60ca18753e306b7a5e5a41ee2d34a80795d8ddc5` |
| `codex/retrieval_matrix_native_analysis/contracts.py` | `98367ef3e91ecae4a7ed2df4c1489320c0e06e1ad9d2eaa2ab57028a0368be25` |
| `codex/retrieval_matrix_native_analysis/launcher.py` | `12b49f16b8b50f92a8ab75973b672435b8c92cfd1da3a0fbcb9f4678b2a97a26` |
| `codex/retrieval_matrix_native_analysis/run.sh` | `31a54ccd58bb298d7cf81d2c06185eab9304d975ff095c50069b55604c4bab38` |
| `codex/retrieval_matrix_native_analysis/source-profiles.json` | `a4374599c3ea3bdea99362ab88abeae8a1b3aab8d9d872cf4f34a91edb34c17b` |
| `codex/retrieval_matrix_native_analysis/synthetic.py` | `cf54f3346fc97c804e06c5cf3b0af2f7248fac9a6f285de03f07de27be3336f1` |
| `codex/retrieval_matrix_native_analysis/test_contracts.py` | `903ca549dd1aee7e436e22787e49b36f6ef5833671f5d0dd955fd77ae9bf0501` |
| `codex/retrieval_matrix_native_analysis/test_full_flow.py` | `615eed510469de0afa87c1d8478e1531f92c206c0971a051a9f13e3c4c1ab945` |
| `codex/retrieval_matrix_native_analysis/test_pipeline.py` | `89ab515801d994d8c9b2e8ed6f40202dff71276db2c9ccea4fbff6d762d65112` |
| `codex/retrieval_matrix_native_analysis/test_workflow.py` | `f82fda7067ea25a4d84a4819de2fb0d503cc8b463d4a8105fe37b48e1f8de105` |
| `codex/retrieval_matrix_native_analysis/workflow.py` | `c60337ffe02837dc3dd7270dd79447a78ea10cff83e4c0c016703ac94a96a7e8` |

shared reader/contracts固定 `7b4855f5d1b7f6e6c1f3850b8a2a9e41b5ac82ed`；SigLIP2 `f3f787eeb193aad8dcc6f2ee43e9609ddf5b47eb`；paired `5675ef59cee99fa6d28e12f07501eeb157c321bd`；native rerank `54f544ce54bca82be3cc8b3c72662b354c6d3e40`。

交接後停止本輪，不啟動新測試，不等待現有全套完成。
