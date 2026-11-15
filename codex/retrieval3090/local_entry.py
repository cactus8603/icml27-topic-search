"""[codex] Relocated development launcher; never manufactures a cluster receipt."""
import json
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
entry, *args = sys.argv[1:]
if entry == 'tests':
    import unittest
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(HERE), 'test_*.py'))
    raise SystemExit(not result.wasSuccessful())
elif entry in {'test', 'analyze', 'analyze2', 'smoke', 'diagnostics'}:
    script = {'test': 'test_analyze.py', 'analyze': 'analyze.py', 'analyze2': 'analysis_v2.py',
              'smoke': 'smoke.py', 'diagnostics': 'diagnostics.py'}[entry]
    sys.argv = [str(HERE / script), *args]
    runpy.run_path(sys.argv[0], run_name='__main__')
elif entry in {'preflight', 'cache', 'cache2', 'supplement', 'benchmark', 'parity', 'platform-cache'}:
    import cache_scores as cache
    score = cache.score
    frozen_preflight = score.preflight

    def local_preflight():
        original = score.original
        receipt = original.DATA.parent.parent / 'runs/retrieval-anchor-rerank/26601/summary.json'
        if receipt.exists():
            # If supplied, an original cluster receipt must pass its exact original check.
            panel, config, snapshots, context = frozen_preflight()
        else:
            # Exact frozen manifest hashes, every JPEG, model config, and counts remain required.
            panel, manifest_sha, coverage = original.preflight()
            config, snapshots = original.models()
            counts = (len(panel['ids']), len(panel['queries']), len({q['group'] for q in panel['queries']}))
            if counts != score.PANEL_COUNTS:
                raise ValueError('Reconstructed exposed panel has different candidate/query/group counts')
            context = {
                'source_job': score.SOURCE_JOB, 'source_commit': score.SOURCE,
                'source_summary_sha256': None,
                'expected_original_source_summary_sha256': score.SOURCE_SUMMARY_SHA,
                'source_receipt_available': False,
                'panel_sha256': manifest_sha, 'coverage': coverage,
                'model_config_sha256': original.MODEL_CONFIG_SHA,
                'depths': score.DEPTHS,
                'revisions': {name: path.name for name, path in snapshots.items()},
            }
        context.update(
            local_runtime_root=os.environ['RETRIEVAL_ROOT'],
            development_launcher='codex/retrieval3090/run.sh',
            provenance_mode='locally-reconstructed-public-exposed-panel',
            original_source_receipt_verified=receipt.exists(),
            scope='local method development only; no sealed independent panel; not formal evidence',
        )
        return panel, config, snapshots, context

    score.preflight = local_preflight
    if entry in {'preflight', 'cache'}:
        sys.argv = [str(HERE / 'cache_scores.py'), '--preflight' if entry == 'preflight' else '--run', *args]
        cache.main()
    else:
        script = {'cache2': 'cache_v2.py', 'supplement': 'supplement.py',
                  'benchmark': 'benchmark.py', 'parity': 'parity_probe.py',
                  'platform-cache': 'platform_cache.py'}[entry]
        sys.argv = [str(HERE / script), *args]
        runpy.run_path(sys.argv[0], run_name='__main__')
else:
    raise SystemExit('Unknown retrieval development command')
