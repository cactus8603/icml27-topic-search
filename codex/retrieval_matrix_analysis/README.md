# [codex] Retrieval matrix analysis

This separate CPU module preserves every frozen producer. Build and independently
review `plan.analysis.review.v1.json`, then freeze its source and role lists before
opening effects. The plan builder reads source plans and prospective paths only.

After the original outputs are all complete, run these as separate actors:

```bash
bash codex/retrieval_matrix_analysis/run.sh mechanical --plan PLAN --freeze-commit FULL_ANALYSIS_COMMIT --job mcq-core2 --actor MECHANICAL_ACTOR
bash codex/retrieval_matrix_analysis/run.sh independent --plan PLAN --freeze-commit FULL_ANALYSIS_COMMIT --job mcq-core2 --actor DIFFERENT_NONIMPLEMENTER --accept-independent-role
```

The wrapper always hides every GPU. Original mechanical reports remain in each
producer's declared `results/`; execution and independent receipts use
`dl/retrieval-matrix-analysis/results/JOB/`. Existing reports or receipts are never
overwritten. Failed partial work requires a separately reviewed recovery decision.
Reading records are appended to this module's preregistration; frozen producer
files remain unchanged. A separate durable journal under
`dl/retrieval-matrix-analysis/reading-anchor/` rejects deleted or rolled-back
reading records. Interrupted journal/log transactions require reviewed recovery
and never reset automatically. The final scientific decision belongs to the user.
