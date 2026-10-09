"""Freeze identities for the explicitly bounded LLMCompass archival milestone.

Run on XMU after isolated-worktree preparation and validation. Only aggregate
manifest data is generated; source edits, Git commits and pushes are separate.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--evidence-repo', type=Path, required=True)
    parser.add_argument('--hardware-root', type=Path, required=True)
    parser.add_argument('--backend-root', type=Path, required=True)
    args = parser.parse_args()
    repo, original = args.repo.resolve(), args.evidence_repo.resolve()
    adapter = repo / 'integration/hbfsim'
    checkpoint = adapter / 'checkpoint'
    names = subprocess.check_output(['git', 'ls-files', 'software_model', 'hardware_model',
        'ae/figure5', 'utils.py'], cwd=repo, text=True).splitlines()
    assert subprocess.run(['git', 'diff', '--exit-code',
        '62321b1ee28ddbdba8a2eb7475d7caa30f75e8be', '--', 'software_model',
        'hardware_model', 'ae/figure5'], cwd=repo, capture_output=True).returncode == 0
    tracked = subprocess.check_output(['git', 'ls-files', 'integration/hbfsim'],
                                      cwd=repo, text=True).splitlines()
    files = [repo / name for name in tracked if name != 'integration/hbfsim/checkpoint/manifest.json']
    assert all(p.is_file() for p in files)
    forbidden = [str(p) for p in files if p.suffix in ('.jsonl', '.ncu-rep', '.nsys-rep', '.log', '.sqlite', '.pyc')
                 or any(part in ('results', '__pycache__', '_build') for part in p.relative_to(adapter).parts)]
    assert not forbidden, forbidden
    assert not any(any('\u4e00' <= char <= '\u9fff' for char in p.read_text(encoding='utf-8'))
                   for p in files if p.suffix in ('.py', '.md', '.json', '.cfg', '.sh', '.html'))
    evidence = original / 'integration/hbfsim/results'
    external = [evidence / 'official-inference-matrix-20261007-r3/manifest.json',
                evidence / 'official-inference-matrix-20261007-r3/validation.json',
                evidence / 'official-inference-matrix-20261007-r3/comparison.json',
                evidence / 'semantic-traffic-20261008-r1/receipt.json',
                evidence / 'semantic-traffic-20261008-r1/semantic-breakdown.json',
                evidence / 'gddr-integration-matrix-20261008-r1/receipt.json',
                evidence / 'gddr-integration-matrix-20261008-r1/validation.json',
                args.hardware_root / 'paired-reference-20261008-r3/collection.json',
                args.backend_root / 'build/hbfsim',
                args.backend_root / 'configs/overlays/dram/rtx4000-ada-gddr6.cfg']
    versions = {}
    for package in ('torch', 'numpy', 'pandas', 'scalesim'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed in manifest-generating interpreter'
    manifest = dict(schema='LLMCOMPASS_ADA_STAGE_ARCHIVE_V1', date='2026-10-09',
        status='PASS_ANALYTICAL_REPRODUCTION_AND_ACCOUNTING_NOT_HARDWARE_ACCURACY',
        branch='codex/llmcompass-ada-stage-acceptance-20261009',
        parent_integration_commit='613cbe73f489ad304b28b10d314c373324cb5dc4',
        official_reference_commit='62321b1ee28ddbdba8a2eb7475d7caa30f75e8be',
        accepted_scope=['unchanged official operator source', 'eight-case analytical operator/phase accounting',
                        'direct mapper boundary and semantic byte conservation', 'offline semantic reporting'],
        excluded_claims=['physical DRAM accuracy', 'all-case timing error within ten percent',
                         'exact native-kernel numerical equivalence', 'continuous GPU cycle simulation',
                         'native GDDR6 command/bank/refresh accuracy', 'all-case GDDR no-regression'],
        hbfsim_in_official_time=False, hardware_accuracy='NOT_ACCEPTED',
        files_sha256={str(p.relative_to(adapter)): sha(p) for p in sorted(files)},
        official_source_sha256={name: sha(repo / name) for name in names},
        external_evidence_sha256={str(p): sha(p) for p in external},
        published_files=len(files), published_bytes=sum(p.stat().st_size for p in files),
        python=sys.version, dependency_versions=versions,
        analytical_dependency_policy='ScaleSim 2.0.2; frozen geometry LUTs; no operator-latency seed',
        validation_commands=[
            'python -B integration/hbfsim/verify_stage_checkpoint.py',
            'python -B -m unittest test_qwen_hbfsim_cosim test_semantic_traffic_breakdown '
            'test_verify_official_inference test_paired_qwen_reference test_evaluate_gddr_integration '
            'test_verify_stage_checkpoint',
            'python -B integration/hbfsim/validate_mapper_accounting.py',
            'python -B integration/hbfsim/validate_mapper_extended.py',
            'python -B integration/hbfsim/validate_full_tensor_views.py'],
        evidence_policy='Published aggregate receipts do not replace external full-ledger measurement evidence.',
        preservation_policy='Copy-only; no original worktree or historical evidence deleted; no force push.',
        subagents=dict(started=0, completed=0, active=0))
    (checkpoint / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'published_files': len(files), 'published_bytes': manifest['published_bytes'],
                      'manifest_sha256': sha(checkpoint / 'manifest.json')}))


if __name__ == '__main__':
    main()
