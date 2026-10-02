"""Detach final validation without retaining SSH pipes."""
from pathlib import Path
import re
import subprocess
import sys

run=sys.argv[1]
if not re.fullmatch(r"[a-zA-Z0-9_-]+",run):
    raise ValueError("Invalid run name")
root=Path(__file__).resolve().parents[2]
run_dir=root.parent/"runs"
pid_file=run_dir/(run+".finalize.pid")
if pid_file.exists():
    raise RuntimeError("Finalizer PID file exists; inspect instead of launching a duplicate")
with (run_dir/(run+"-finalize.log")).open("a",encoding="utf-8") as log:
    process=subprocess.Popen([sys.executable,str(Path(__file__).with_name("finalize_recovery.py")),run,*sys.argv[2:]],
        cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pid_file.write_text(str(process.pid)+"\n",encoding="utf-8")
print(f"Finalizer PID: {process.pid}")
