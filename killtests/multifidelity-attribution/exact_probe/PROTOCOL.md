# [codex] 新查詢精確歸因診斷

本探針只分析上一輪十方向混合在新查詢上失效的原因。原 Stage 0、跨種子資格、停止條件各沿用原判準。前一輪已看過全部 350 目標查詢的彙總，故本次仍屬探索性。

## 固定資料與量測

以 PCG64 seed `7611402` 在既有目標池的位置 50–224 和 225–399，各均勻抽一個合法的連續六欄起點，得到 `[118,124)` 和 `[268,274)`，共 12 個新精確查詢。抽樣依位置與獨立種子，未讀個別目標分數。無替補。沿用 ResNet-9 seed 0、CIFAR-10、原設定、query seed、50,000 個訓練列與原 200 個 1% 移除遮罩。以原 `exact_y.py` 的正向重播恢復中間狀態並反向微分，每區塊一次。計算量包含原訓練軌跡的 seed 0 正向重播，無新模型設定、資料、移除重訓或 seed。使用獨立 `exact_probe/<run-id>` 輸出。

沿用上次分析的 $U,D,C,\alpha$ 構造。前 50 個查詢的 Gram 取前十特徵向量 $U$，精確方向為 $A=Y_{cal}U$。便宜方法 C 只由前 50 查詢、遮罩 0–99 的 LDS 選定。$\alpha=\langle A,C_{cal}U\rangle/\|C_{cal}U\|^2$ 用全部訓練列。診斷用的 12 欄新 $Y_t$ 僅供 oracle 與評估，原混合與同預算 Spell 都只有 A 這 10 個精確方向。遮罩 0–99 校準診斷用 oracle，100–199 作固定評估，遮罩內跨方法成對。

## 公式與對照

- $P_A$ 是 A 在 50,000 維樣本權重空間中的正交投影，使用 float64 SVD，奇異值相對門檻 `1e-10`。秩不足十即停止。逐查詢報 $\|P_A Y_t\|^2/\|Y_t\|^2$ 與測試遮罩的中心化預測 $R^2$。第一量是原始權重歐氏度量下的線性投影能量，精確等於該度量的最佳線性重建比例。第二量衡量遮罩上的線性誤差，可能為負。LDS 另外以實測重訓分數的排名計算。
- $M$ 的元素是遮罩的負指示值。精確預測為 $E=MY_t$，原 Spell 延伸為 $S=MA D$，固定補值為 $R=\alpha M(C_t-C_{cal}UD)$，原混合為 $S+R$。對照報七種純便宜方法及同預算 S。$P=MP_A Y_t$ 是對精確向量的診斷 oracle，$P+R$ 保留補值並替換 S，以隔離 Gram 傳遞。
- 以 $F=MQ$（Q 為 A 的正交基底）在遮罩 0–99 對 E 做逐查詢截距與最小平方，測試遮罩上算 $F\hat b$（截距保留以計算線性誤差）與 $F\hat b+R$。此項使用新 $Y_t$，以遮罩 0–99 的精確預測差及線性誤差擬合。
- 尺度診斷在遮罩 0–99 對中心化 P 與 S 擬合一個逐查詢非負比例 β，報 `βS+R`。非負係數固定符號。β 與線性空間擬合只用 0–99 的 E 或 P。P 的權重投影本身是全精確向量的 oracle，所有 oracle 都需新 $Y_t$。實測重訓損失僅用於評估和依原規格選 C。
- 報 `S` 對 `E` 及 `P` 的留出遮罩 Spearman、中心化 $R^2$、預測標準差和 β，另報全精確與各方法的逐查詢實測重訓 LDS（以保存的交叉熵為真值）。`R² = 1−sum(((E−mean(E))−(prediction−mean(prediction)))²)/sum((E−mean(E))²)`，因此只衡量遮罩間變化。clip 不用於 R²。區間只採查詢配對 bootstrap 10,000 次 seed `7611403`，另列兩個採樣區塊結果，CI 為描述，因僅兩塊且共享訓練軌跡與遮罩。

## 事前判讀

輸入不符、訓練重播基準不符、測得方向退化、校準設計秩不足、非有限、或任一必要預測在可評分查詢為常數，標記未能測試。重播基準檢查目標 query loss 與原主 checkpoint 的 RMSE 除以 `max(std(checkpoint_probe_losses), 0.1)` 小於 `0.02`，逐區塊及整組最大誤差另外記錄。精確 LDS 平均必須至少 0.88 才給以下機制標記。若存檔重訓真值恒定，各方法共同排除該欄並保存全組未能測試；至少九欄可評分才給描述機制標記。所有門檻是診斷規則，無 Stage 0 效力。

以 `max(S 的 LDS, 七種便宜方法的 LDS)` 為最強基線，+0.04 作先前設定的實用參照。以下標記可並存，其他情況保留未能定位：

- **方向覆蓋偏低**：遮罩校準 span oracle 的測試 $R^2$ 中位數 <0.25，且權重投影及遮罩校準 oracle 混合皆未達基線 +0.04。此標記限定權重及遮罩上的線性度量。
- **傳遞有改善空間**：`P+R` 同時比原混合及最強基線高至少 0.04。若只有遮罩校準 oracle 達到該標準，單列「遮罩度量擬合有改善空間」。
- **單純尺度有改善空間**：`βS+R` 同時比原混合及最強基線高至少 0.04。即使成立，比例仍藉由新精確欄擬合，後續需單獨驗證可取得的尺度代理。

所有區塊沿用單一原始訓練軌跡，真實新資料或不同訓練 seed 的泛化不在此診斷範圍。報告以通過輸入檢查的機械輸出為準，新增 GPU 或門檻調整須另行協調。

## 由 repo 根目錄執行

`plan <run-id>` 為唯讀，列出 CIFAR-10 索引，`self-test <未存在的測試目錄>` 只使用合成資料。正式執行由協調者提供已凍結 commit 並登記兩項 Slurm 作業。`prepare` 建立 `/work/ChihChungHsu/cactus8603/icml/runs/multifid/exact_probe/<run-id>` 和獨立 logs 目錄，需要能寫入共用的 run 根目錄。

```bash
probe_dir=killtests/multifidelity-attribution/exact_probe
probe_python=/work/ChihChungHsu/cactus8603/icml/envs/attrib-jax/bin/python
probe_commit=$(git rev-parse HEAD)
probe_run=target12-v1
PYTHONDONTWRITEBYTECODE=1 "$probe_python" "$probe_dir/probe.py" prepare "$probe_run" "$probe_commit"
probe_job=$(sbatch --parsable "$probe_dir/exact.sbatch" "$probe_run" "$probe_commit")
sbatch --dependency="afterok:${probe_job%%;*}" "$probe_dir/score.sbatch" "$probe_run" "$probe_commit"
```

GPU 陣列一次執行一個工作，每工作 45 分鐘上限。原 10 查詢區塊約 11 分鐘，本次兩個 6 查詢區塊合計預估 20–30 GPU 分鐘。CPU 工作只讀完整的兩個收據，寫入隔離 run 的 `results.json`。兩階段對照 Git 凍結檔案內容、資料雜湊及 run manifest。執行 GPU 的原始腳本及其本地模組先複製到各區塊的獨立目錄，收據保留原始輸出及配置雜湊。每個 run ID 是一次嘗試，任何重試用新 ID 並保留失敗資料。
