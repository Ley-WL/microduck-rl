"""Reproducible CPU preview disturbance battery; no robot connections."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
preview = Path(__file__).with_name("preview_hd1910.py")
cases = [(s,d,False) for s in (.08,.15,.30,.50) for d in ("forward","backward","left","right")]
cases += [(.30,d,False) for d in ("up","down")]
cases += [(.30,"forward",True),(.30,"backward",True)]
results = []
for strength,direction,foot_only in cases:
    command = [sys.executable,str(preview),"--headless","--seconds","8",
               "--push-at","2","--push-strength",str(strength),"--push-direction",direction]
    if foot_only:
        command.append("--foot-only")
    run = subprocess.run(command, capture_output=True, text=True, check=True)
    records = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    metrics = records[0] | records[2]
    metrics["fell"] = metrics["maxTiltDeg"] > 45
    metrics["footOnly"] = foot_only
    results.append(metrics)
    print(json.dumps(metrics),flush=True)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(results,indent=2),encoding="utf-8")
