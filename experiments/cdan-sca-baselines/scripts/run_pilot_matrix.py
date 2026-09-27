#!/usr/bin/env python3
"""Run the frozen Source Only/DANN pilot matrix and write a compact summary."""
from __future__ import annotations
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/results/cdan-sca-baselines"
PYTHON = str(Path.home() / "venvs/tf/bin/python")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in ("source_only", "dann"):
        for seed in (42, 43, 44):
            log = OUT / f"pilot_{method}_seed{seed}.log"
            cmd = [PYTHON, "experiments/cdan-sca-baselines/scripts/run_baseline_smoke.py", "--method", method, "--epochs", "10", "--limit-source", "5000", "--limit-target", "5000", "--limit-eval", "2000", "--batch-size", "64", "--seed", str(seed)]
            with log.open("w") as handle:
                subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            records = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{"epoch"')]
            rows.append({"method": method, "seed": seed, **records[-1]})
    summary = {"config": {"epochs": 10, "limit_source": 5000, "limit_target": 5000, "limit_eval": 2000, "batch_size": 64, "seeds": [42, 43, 44]}, "rows": rows}
    (OUT / "pilot_matrix_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
