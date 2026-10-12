"""Detach this run's regression checks and retain PID/log for inspection."""
import re,subprocess,sys
from pathlib import Path
run=sys.argv[1]
if not re.fullmatch(r'[A-Za-z0-9_-]+',run):raise ValueError('Invalid run')
root=Path(__file__).resolve().parents[2];runs=root.parent/'runs'
pid=runs/(run+'.guard.pid')
if pid.exists():raise RuntimeError('Guard already launched')
with (runs/(run+'-guard.log')).open('w') as log:
    child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('guard_gait_repair.py')),
        run,*sys.argv[2:]],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pid.write_text(str(child.pid));print('Guard PID:',child.pid)
