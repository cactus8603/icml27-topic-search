# CLIP 代理修復候選證據

`review_evidence.py` 只讀 26516 彙總、Slurm 記錄及兩個固定 Git 版本，重算同一代理門檻並輸出候選修復證據。三組按凍結程式做完整同 seed 範圍檢查。彙總只有計數，獨立核對的範圍為工作、來源校驗、覆蓋收據與計數可行性。原 Gate B 判讀由原流程管理。

```bash
stage=codex/experiments/t2i-gate-repair
python3 -B "$stage/review_evidence.py"
python3 -B "$stage/test_review_evidence.py"
```
