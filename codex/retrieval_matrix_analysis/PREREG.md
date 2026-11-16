# [codex] Retrieval matrix: CPU analysis and independent verification v1

[codex] 2026-11-16. This analysis is specified before opening any main model score,
report, aggregate or verdict from the jobs below. No new GPU inference is allowed.
The original producer preregistrations, plans, code and checkpoints remain immutable.

## Order and scope

The mandatory order is the original frozen mechanical analysis script, a different
nonimplementer's independent recomputation of the original arrays, and the user's
final decision. A recomputation PASS means numerical agreement only. It never
stands in for empirical P0 qualification or the user's decision. The launcher
cannot emit a user decision. Every analysis plan and this document must be
reviewed and committed before an effect file is opened.

The first supported producers are CORE2/8 MCQ at
25f7bd515875a0c698202d0047d7b7952fd6846b. Their scores reside in each model's
`main/` directory. The same plan explicitly lists the six corrected native
embedding and six native MCMR jobs at
99431594893bba94337b3befd10a7a483ba7374f. Original failed versions are not adopted.
No result file is read to construct the plan or the CPU tests.

## Frozen numerical interpretation

CORE MCQ retains all 15,912 original examples, 58,891 pairs and eight subsets.
COLA uses the original text-choice axes and strict comparisons, SCPP the equal
macro over its five original subsets, and NegBench the equal COCO/VOC macro.
Ties fail. All three benchmarks remain visible. At least two distinct official
benchmarks must be within the original absolute tolerance of .01. The original
acceptance values are taken from the frozen producer plan without revision.
No row deduplication, substitute benchmark or pooled row-weighted macro is allowed.

Native embedding P0 retains COCO, Flickr and NegBench together with every original
query, including the documented empty captions. Only native BF16 GPU score caches
and original GPU `topk` admission define the official metrics. Secondary FP32
stable full-gallery ranks cannot qualify a model. Independent CPU scoring keeps
the original tied boundary members and applies the original evaluator's reverse
lexical document-ID tie ordering. I2T recall is relevant-caption fraction. Binary
AP@10 divides by all relevant documents, including positives beyond the cutoff.
All seven reported IR fields in both directions are recomputed. Native GPU dot
products are audited through exact source, item/batch fingerprints and byte
receipts; no new CPU or GPU dot product is presented as identical inference.

Native MCMR retains 104,981 distinct product IDs, their complete renderer text
and 3,997 queries. Products sharing an image are not merged. Official native
top100 supports fractional recall, nDCG/AP and MRR@10. MRR uses the native first10
admission followed by evaluator score/document-ID ordering. Ordinal diagnostics
remain separate. The verified MMEB renderer differs from the unavailable original
CORE converter: MCMR is descriptive, never strict CORE P0 qualification.

Agreement with original mechanical numbers requires absolute difference at most
1e-12. Incomplete, nonfinite, changed-model, changed-source, changed-array,
changed-data, wrong-reviewer or wrong-denominator evidence fails closed. A failed
or partially written mechanical stage is preserved; there is no automatic retry,
report overwrite, numerical fallback or model selection based on outcomes.

## Roles and reading records

The plan lists producing implementation authors, analysis authors and the central
producer identity for each job. The independent reviewer must differ from all of
them and from the mechanical actor, and explicitly attest to that role. Review
those role lists before freeze. Shared model/backend implementations count as
producer implementations for MCMR as well. Qwen2 runtime-fix authors remain on
both embedding and MCMR author lists. Plan/ref/model/output paths are checked
against the original central schedule, not just its checksum. The analysis implementation author does not
perform the formal independent review. Array agreement does not imply adequate
statistical power, generalization or scientific significance.

The immutable body ends at the marker below. After freeze only machine-generated
`[codex]` JSON reading records may be appended here. Before any score, report,
completion or original index receipt is opened, the launcher durably appends
READ_STARTED with actor, job, complete prospective path list, analysis-plan SHA
and source commit. A terminal event appends the actual available file hashes.
Records form a hash chain. Per-file hash failures append READ_FAILED with the
affected paths; only a successful completed mechanical reading may precede
independent review. No outcome number is placed in this document. An
unfinished start remains visible after a crash. Rewriting the body, deleting or
editing an earlier event, replacing the plan or adding arbitrary prose requires
a new reviewed analysis version. The original producers' preregistrations are
not modified by this reading log.

A separate append-only journal under
`dl/retrieval-matrix-analysis/reading-anchor/` durably anchors the complete log's
byte length, SHA, event count and latest event hash. Its genesis binds this
preregistration's path/body, the analysis plan SHA and analysis source commit.
Only a genuinely empty log may initialize a missing anchor; an existing log
without its anchor is rejected. Before each log append, a PREPARE entry containing
the exact expected next event/head/length is written and fsynced to the journal.
Then the preregistration append is fsynced, then a COMMIT entry is fsynced. A new
journal and its containing directory are both fsynced. Artifact reading cannot
start until all steps succeed. Every append and source gate compares the full
log with the durable journal: tail deletion, full-tail deletion, earlier-prefix
rollback or disagreement is rejected. An interrupted transaction fails closed
and requires reviewed recovery; no source gate or append resets, repairs or
adopts it automatically. Synthetic tests use isolated temporary anchors.

<!-- APPEND-ONLY READ LOG -->
