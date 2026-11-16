# ICML 2027 選題與實驗

我們用小規模實驗確認研究現象、可用訊號，以及相較現有方法的改善空間。點進題目資料夾可看介紹、設計、結果、程式，以及資料集、環境和執行方式。

本 repo 是**中樞**：放協作規則、選題、GPU 送件紀錄與跨題目報告。每個題目另有題目 repo（`icml27-<題目>`）。標「已搬」的題目以題目 repo 為正本，本 repo 的 `topics/` 只是封存；標「遷移中」的題目已在題目 repo 建立唯讀鏡像，但目前仍以本 repo 為正本；這一輪工作結束後用 `scripts/topic_repo_sync.py` 合併並切換。規則見 [協作規則第 0 節](COORDINATION.md#0-repo-結構中樞與題目-repo)。

## 優先完成現象與訊號

- [用訓練前的 Λ 預測防禦漏出](topics/inoculation-lambda/)　[repo `icml27-inoculation-lambda`](https://github.com/cactus8603/icml27-inoculation-lambda)（遷移中）　補完四類防禦的漏出差異，檢查 Λ 能否預測哪種設定比較容易漏出。
- [文生圖罕見失敗的機率估計](topics/t2i-rare-failure/)　[repo `icml27-t2i-rare-failure`](https://github.com/cactus8603/icml27-t2i-rare-failure)（遷移中）　補完正式事件率和人工盲評，確認生成失敗有多罕見、偵測器報出的陽性有多少是真的。

## 其他候選

- [多條件影像檢索](topics/compositional-retrieval/)　[repo `icml27-compositional-retrieval`](https://github.com/cactus8603/icml27-compositional-retrieval)（遷移中）
- [用少量精確計算改善訓練資料歸因](topics/multifidelity-attribution/)　[repo `icml27-multifidelity-attribution`](https://github.com/cactus8603/icml27-multifidelity-attribution)（遷移中）
- [只移植少數 RL 推理決策的蒸餾](topics/rl-sparse-distill/)　[repo `icml27-rl-sparse-distill`](https://github.com/cactus8603/icml27-rl-sparse-distill)（已搬）
- [強驗證器與少量金標](topics/fallible-verifier/)　[repo `icml27-fallible-verifier`](https://github.com/cactus8603/icml27-fallible-verifier)（已搬）

## 暫停與淘汰

- [大型平行解碼提交集合](topics/large-commit-sets/)　[repo `icml27-large-commit-sets`](https://github.com/cactus8603/icml27-large-commit-sets)（已搬）　論文潛力降為 **2/5**，暫停。
- [注意力寫入與最佳化器](topics/gate-optimizer.md)　論文潛力 **2/5**，本輪淘汰。
- [條件擴散的兩個時鐘](topics/biased-gen-cfg.md)　論文潛力 **2/5**，本輪淘汰。

[選題標準](criteria.md) · [研究排程](shortlist.md) · [實驗硬體](codex/HARDWARE.md) · [作業紀錄](killtests/SUBMITTED.log)
