# Gate (b) 原種子的 CLIP 精度修復驗證

此驗證用完整原種子檢查修復後的 CLIP 平滑代理，並讓獨立審查能從逐種子分數重算原門檻。新種子精度重驗 `26516` 提供修復依據。歷史 `26289` 與其正式判讀留在原路徑。這次的候選收據供獨立審查及定案，後續 Stage 1 仍依原程序。 [原預登記](../../../killtests/t2i-rare-failure/PREREG.md) · [新種子重驗](../t2i-gate-b-precision/PREREG.md)

## 固定面板與算術

- SD v1.4、E1／N1／B1，各用 `[0, 10000)`。封存原 `f54cb310a921966440754d42f65f86d9f565dc62` 生成入口。每 500 種子分塊、分塊內每批 32 張，包含各分塊最後 20 張。DDIM 25 步、512×512、guidance 7.5、eta 0、FP16 及逐種子 CPU 雜訊沿用。保持原 Gate (a) 與 bench 再驗順序，並在 CPU 預檢和實際執行前核對兩者。原 Gate (b) 的門檻與五個配置讀自 `37e6ae9a63cd3035822dccca2acf90ef12efec93` 封存。原 CPU smoke 有既有紀錄，此次另以合成輸入測試修復接線與驗證器。 [原 CPU smoke 紀錄](../../../killtests/t2i-rare-failure/logs/smoke_regen.log)
- 僅注入 `2b1c7f64b8b9013a9a32f8353024c1d6f82b6e53` 已驗證的 CLIP 算術。前處理、文字與影像特徵、正規化和 scale 指數維持原精度。最後的 scale 乘法、點積與 softmax 改以 FP32，點積期間關閉 TF32 並回復原設定。原生成入口核對離線 SD 和 CLIP revisions。沿用已核過的 torch、diffusers、transformers 版本。
- 原驗證器重新核對門檻 `0.5／0.9／0.99／0.999`、每提示 10,000 個種子、機率範圍及完整來源。任一提示至少 1,000 個相異分數，且任一門檻命中數達 100、比率至多 5%，候選重播即符合原代理計數條件。

## 逐塊驗證與保留

- 每塊保留 INT64 seed、FP32 CLIP 機率與原格式所需的兩個 NaN 欄。保留原 sidecar 的配置、執行版本、作業編號，另核凍結 commit、兩個 checkpoint revisions、FP32 修復標記、原生成程式雜湊與 batch size。逐塊驗收固定檔名、種子順序、全部 60 個資料與 sidecar 對。影像只在記憶體內由原生成器傳給 CLIP。
- 結果在 `icml/runs/t2irare/gate-b-original-replay/<job>/`。原驗證器將全部分數與 sidecar SHA-256 納入 `source_sha256`。完整驗收才另發 `precision_replay_verified.json`，內含凍結來源、Gate (a) 與 bench 證據及各門檻計數。Slurm 成功表示完成驗收，代理判讀見收據的 `status`。獨立複核者在作業完成後重算並核對收據。中斷作業保留原目錄，重送使用新作業路徑。
- 一張 H200、16 CPU、64 GiB RAM，Slurm 上限 3 小時。以 `26516` 同 30,000 張的成本推估約 1.5 小時。純陣列未壓縮約 0.6 MB，含壓縮封包與收據預估低於 2 MiB。

```bash
stage=codex/experiments/t2i-gate-b-original-replay
bash "$stage/run.sbatch" "$freeze" --check-only
sbatch "$stage/run.sbatch" "$freeze"
bash "$stage/run.sbatch" "$freeze" --verify-job "$job"
```
