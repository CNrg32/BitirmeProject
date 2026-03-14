#!/usr/bin/env python3
"""
Run pytest and optionally update reports/test_plan_report.md Result column from results.
Usage: PYTHONPATH=src python scripts/run_tests_and_update_report.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "test_plan_report.md"


def run_pytest() -> tuple[int, bool]:
    """Run pytest; return (exit_code, all_passed)."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=no", "-q"],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    out = (r.stdout or "") + (r.stderr or "")
    all_passed = "failed" not in out.lower() and r.returncode == 0
    return r.returncode, all_passed


def update_report_results(all_passed: bool) -> None:
    """Set all Result cells in the Test Cases table to Passed or Failed."""
    if not REPORT.exists():
        return
    text = REPORT.read_text(encoding="utf-8")
    # Table rows with | Passed | or | Failed |
    pattern = r"(\|\s*)(Passed|Failed)(\s*\|\s*$)"
    replacement = r"\1Passed\3" if all_passed else r"\1Failed\3"
    # Only replace in the Test Cases section (between ## 6 and ## 7)
    parts = re.split(r"(## 6\) Test Cases.*?)(## 7\))", text, flags=re.DOTALL)
    if len(parts) >= 3:
        mid = re.sub(pattern, replacement, parts[1], flags=re.MULTILINE)
        new_text = parts[0] + mid + parts[2]
        REPORT.write_text(new_text, encoding="utf-8")


if __name__ == "__main__":
    code, passed = run_pytest()
    update_report_results(passed)
    sys.exit(code)
