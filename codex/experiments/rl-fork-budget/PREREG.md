# 同一單 token 分叉的 3,000 與 3,710 token 對照

[codex] 2026-11-15（台北）。配對 A 在 3,000 token 的既有分叉量到正確率差，也看到兩臂截斷率大幅不同。這次用新題和同一條續寫在兩個預算評分，檢查較長預算下，仍截斷的回答能否解釋所見差距。適用於共同前綴、單步介入及由同一 Base 貪婪續寫的局部效果。正式配對資格、多次取樣以及 WiSE-FT 仍依原設計。 [原分叉](../rl-single-token-fork/PREREG.md) · [正式預登記](../../../killtests/rl-sparse-distill/PREREG.md)

## 固定輸入與介入

- 沿用原分叉的 MATH test 來源、題文正規化、去重、排除、雜湊排序及標答抽取，不更換資料集。保留原排序中的位置 128 至 255（從零起算）。以原 128 題指紋驗證分界。新 128 題及標答 SHA-256 為 `64e98dc13a33fcc78658a658adb7df6c9952347f9c39717b59288366581333f7`，兩模型相同的提示 token 陣列 SHA-256 為 `37b32de4ddabfb2e295d42ae741a6e952abccfee2878f63903a2505784005603`。去重依題文，未辨識改寫或訓練曝光。
- 固定原 Qwen2.5-Math-1.5B Base、Oat-Zero、模板、兩權重指紋、原評分、BF16、PyTorch 2.5.1、Transformers 4.48.2 及 SDPA 數學核心。沿 Base 貪婪路徑找前 256 步首次合格的非停止分歧。兩臂各強制自己的 argmax token，再由同一 Base 貪婪續寫。未分叉的題留於分母，配對差為零。 [權重資產與停止條件](../rl-single-token-fork/PREREG.md#固定題組和權重)
- 兩個 checkpoint 均為 4,096 context。本題組最長提示 386 tokens，因此所有題、兩臂同樣允許產生 3,710 個 token。每臂只生成一次，遇到停止 ID 或共同最長預算即停止。同一序列在 3,000 與 3,710 token 各切一次，含共同前綴和先前強制的 token。在切點當步已自然停止者沿用同一回覆的標籤，尚未自然停止者依原規格算錯。兩個預算共享 3,000 token 以前的軌跡，分段批次另外核對 padding 與有效 context 界。

## 判讀

- 主要結果是長預算「RL token 減 Base token」的正確率差，並給仍截斷樣本的二元結果識別界。對每題的差值，下界將仍截斷的 Base 臂視為可能答對、仍截斷的 RL 臂視為錯，上界反過來。正常終止使用既有正確標籤。這兩條界量未解結果最多能改變多少差距，並保留 128 題完整分母。
- 以題配對重抽 10,000 次、PCG64 seed `2026111521`，使用同組重抽量兩個預算的差、其差中差及上下界的描述性雙側 95% 百分位區間。長預算保守下界點估至少 +5 個百分點且該下界的區間下端大於零，記 `advantage_robust_to_censoring`。長預算樂觀上界的區間上端低於 +5 點，記 `advantage_below_reference`。其餘記 `inconclusive`。+5 點是本次機制訊號的實用參考，取自原主實驗對 pass@1 增益的幅度門檻。短預算同場差與截斷數提供對照。
- 產物保留各預算的逐題停止 ID、長度、正確性、分叉位置、所選 token 和前綴指紋。生成文字只在評分時留於記憶體。正式結果保存在 `icml/runs/rsd-fork-budget/<job>.json`，待依協作規則登記及獨立複核。

## 執行

CPU 預檢核對新舊題組、模型與依賴、全部 128 份標答自配，並以合成測試覆蓋停止當步、重截、識別界和三態判讀。兩個合成題先於相同凍結版本跑 GPU 續寫和重截，正式診斷須持相依的技術憑證。一張 H200，每次正式診斷上限 90 分鐘。先前 3,000-token 診斷實測 31 分鐘；本次預算提高約 24%，顯存與耗時待驗。依協作規則於凍結與獨立審查通過後送出。

```bash
stage=codex/experiments/rl-fork-budget
freeze=REVIEWED_FULL_COMMIT
bash "$stage/run.sbatch" "$freeze" --check-only
smoke_job=$(sbatch --parsable --time=00:15:00 "$stage/run.sbatch" "$freeze" smoke)
smoke_job=${smoke_job%%;*}
sbatch --parsable --dependency="afterok:$smoke_job" "$stage/run.sbatch" "$freeze" diagnostic "$smoke_job"
```
