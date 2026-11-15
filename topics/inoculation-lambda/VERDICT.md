# 用訓練前的 Λ 預測防禦漏出｜目前結果

**目前現象。** 語言盲評中，無防禦的 20 筆回覆有 17 筆含西語，基礎模型和加防禦各為 0/20。各臂 200 筆的離線檢查也見到語言分離。這批結果支持語言漏出，原三任務的重現與跨初始化信度仍待正式驗證。合成 GCD 五臂已完成獨立複核，回答格式造成大幅不確定，各臂之間可重複的漏出差異未建立。 [語言檢查](code/experiments/lambda-language-local/PREREG.md#讀取紀錄) · [GCD 五臂實測](../../codex/reports/2026-11-15-gcd-pilot-verdict-codex.md)

**目前訊號。** 一步近似已通過數值檢查。GCD 五臂的訓練後漏出區間互相重疊，訊號方向未定。四家族的 **51/51 組事前評分**及 PHM 基線已完成。正式行為第一批在模型存檔驗收遇到問題。修復已通過鎖版測試與原清冊檢查，重跑作業 `27112` 已排隊。等行為測量完成，才會比較留出家族的排名與設定選擇。 [GCD 五臂實測](../../codex/reports/2026-11-15-gcd-pilot-verdict-codex.md) · [四家族設計](code/experiments/lambda-gcd-family-design/PROTOCOL.md) · [作業紀錄](../../killtests/SUBMITTED.log)

**目前空間。** 還缺重複訓練的信度上限，以及 PHM 等方法在同一批設定上的表現。知道這兩者，才有尺度判斷預測漏出還能改善多少。 [比較規則](code/experiments/lambda-gcd-family-design/evaluation/README.md)

## 信心、潛力與資源

- **現有證據 2/5，條件式論文潛力 4/5。** 已看到部分漏出現象。若訓練前分數能跨家族選出低漏出設定，還能在其他任務重現，就有清楚的研究主張。
- **下一步資源估計。** 四家族的訓練前評分與 51 組行為試驗合計約 **5.3–7.0 H200 GPU 小時**，PHM 基線另設 30 分鐘上限。兩組主要作業各同時用 1 張 H200，依序執行約 **5.5–8 小時**，含 CPU 彙整，不含排隊。本次正式 GPU 作業每格申請 **170 GiB RAM、16 核 CPU**，只長期保留數字及收據。 [訓練前評分資源](code/experiments/lambda-gcd-family-design/signals/README.md) · [四家族運算估算](code/experiments/lambda-gcd-family-design/PROTOCOL.md) · [行為試驗配置](code/experiments/lambda-gcd-family-design/outcomes/run.sbatch) · [PHM 配置](code/experiments/lambda-gcd-family-design/phm/run.sbatch)

## 深入調查

既有研究已用訓練前的梯度診斷一類防禦。Persona Hierarchy Model（PHM）也用模型對訓練情境與一般情境的內部表示，預測學到的行為會不會外溢。

我們要把整類防禦留作獨立測試，看 Λ 能否比這些基線更常選到實際漏出較少的設定。 [防禦漏出文獻深查](../../reports/2026-11-14-54-deepcheck-inoculation-leakage.md)

[回題目首頁](README.md)
