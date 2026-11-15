# 資料集、環境與執行方式

以下路徑的 `$ICML` 是 H200 叢集上的 `/work/ChihChungHsu/cactus8603/icml`。程式與設定在 [本題 code 目錄](code/)，原始資料和權重由列出的叢集路徑讀取。

## 資料集與模型

樣本由固定提示和種子生成。原代理檢查用三個提示各 10,000 個種子。正式首段是 6 個模型 × 14 個提示，每格 2,000 張，共 168,000 張。第二段按凍結規則，先選至多 12 格補到 20,000 張，再選至多 6 格補到 100,000 張。人工盲評至少包含 50 張罕見陽性及正、負對照各 50 張。

六個模型為 Stable Diffusion v1.4、UCE、ESD、SalUn、RECE 和 AdvUnlearn。模型及偵測器使用 `$ICML/hf_home/hub` 的固定版本。前置檢查用 CLIP，正式事件由 NudeNet 和 Falconsai 一起判定。NudeNet 的 `320n.onnx` 隨套件安裝。模型修訂、偵測器門檻及樣本設定可在程式旁查閱。[模型設定](code/killtest/configs/models.json) · [偵測器設定](code/killtest/configs/detectors.json) · [取樣設定](code/killtest/configs/sampling.json)

## 環境

使用 `envs/t2irare/bin/python`。Python 3.12.3、Torch 2.5.1、Diffusers 0.31.0、Transformers 4.46.3、NumPy 1.26.4、NudeNet 3.4.2、ONNX Runtime 1.32.0。預檢需要可寫的 `$ICML/tmp/t2irare`、原始 Git 提交，以及內部快取中與固定版本一致的 `refs/main`。正式生成用 H200，NudeNet 與盲評結算走 CPU。

## 怎麼跑

~~~bash
bash scripts/check-frozen.sh 45fc76383dd5cfbf9eb016e514ee9d6ac80168fb \
  codex/experiments/t2i-requeue-accounts/run.sh --check-only
sacct -X -n -P -j 26765,27116,27117,27161,27190 --format=JobID,State,ExitCode
~~~

正式取樣需要已通過並驗證的 `26663/precision_replay_verified.json`。按已登記流程完成首段驗收，依選格檔的雜湊補樣。發生叢集重排時，驗收從每次配置計入 GPU 時間。完成正式取樣後整理盲評包，再依規則計算陽性可信度。[正式取樣與盲評操作](code/experiments/t2i-formal-replay/PREREG.md) · [重排後的選格入口](code/experiments/t2i-requeue-accounts/PREREG.md)

輸出在 `$ICML/runs/t2irare/formal-replay/4351ebb70f2e7ad72e0ec8b99658d9ed41c06567/`，分為 `scores/`、`operations/`、`configs/` 和 `review/`。日誌在 `$ICML/tmp/t2irare/`。給標註者的盲評包只有圖片、編號與標籤欄。

[CPU 預檢工具](../../scripts/check-frozen.sh)從 Git repo 根目錄讀取指定提交，在叢集暫存區建立原路徑的執行副本。需要完整 Git 歷史和已備妥的資料、環境。正式 GPU 作業沿用對應預登記指定的執行工作區與凍結提交，依[協作規則](../../COORDINATION.md)提交及登記。

[回題目首頁](README.md)
