import argparse
from pathlib import Path
import re
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--kind', choices=['getup', 'sitstand'], required=True)
p.add_argument('--gpu', required=True)
p.add_argument('--iterations', type=int, required=True)
a = p.parse_args()
if not re.fullmatch(r'[A-Za-z0-9_-]+', a.run):
    raise ValueError('Invalid run ID')
root = Path(__file__).resolve().parents[2]
run_dir = root.parent/'runs'
pid_file = run_dir/(a.run+'.watch.pid')
if pid_file.exists():
    raise RuntimeError('Watcher PID file exists; inspect the process before retrying')
with (run_dir/(a.run+'-watch.log')).open('a') as log:
    process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('watch_skills.py')), *sys.argv[1:]],
                               cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
pid_file.write_text(str(process.pid)+'\n')
print('Watcher PID:', process.pid)
