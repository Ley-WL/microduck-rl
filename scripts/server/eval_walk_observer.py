"""Evaluate the newly trained export with the portable 30-command simulation."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence"

def main():
    exports = list((OUT / "main").rglob("*.onnx"))
    assert len(exports) == 1, f"Expected one final export, found {exports}"
    env = os.environ.copy()
    env.update(WALK_OUTPUT_DIR=str(OUT / "input-response" / "observer5000"),
               WALK_POLICY_PATH=str(exports[0]))
    subprocess.run([sys.executable, str(ROOT / "scripts/compare_walk_inputs.py"), "observer5000"],
                   cwd=ROOT, env=env, check=True)
    (OUT / "evaluation-passed.json").write_text(json.dumps({
        "models": ["observer5000"], "cases": 30, "worldsPerCase": 16,
        "accepted": False, "note": "Simulation completed; inspect response before acceptance."
    }, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
