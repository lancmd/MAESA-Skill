"""Smoke-test the reviewer-evidence package on an empty project."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
workspace = ROOT / "outputs" / "review_evidence_smoke_project"
workspace.mkdir(parents=True, exist_ok=True)
result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "build_review_evidence.py"), "--workspace", str(workspace)],
    check=True, capture_output=True, text=True,
)
payload = json.loads(result.stdout)
assert payload["overall_status"] == "pending_validation"
assert payload["evidence"]["classification_independent_validation"]["status"] == "not_computable"
assert payload["evidence"]["plus_historical_backcast"]["status"] == "not_computable"
assert (workspace / "最终成果" / "期刊论文" / "同行评议数据核查" / "独立验证样本模板.csv").is_file()
print("review evidence smoke: ok")
