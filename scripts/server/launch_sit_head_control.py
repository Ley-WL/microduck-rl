import subprocess,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence/head-control';OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'launch-v3.json').exists(),'Track existing launch'
files=[ROOT/'scripts/server/run_sit_head_control.py',ROOT/'src/mjlab_microduck/tasks/microduck_hd1910_sit_head_env_cfg.py',ROOT/'logs/rsl_rl/hd1910_sit_head_mechanical/head_seed/seed.pt',ROOT/'evidence/seed-audit.json',ROOT/'scripts/evaluate_sit_head_cpu.py',ROOT/'scripts/bam_cpu_controller.py']
with (OUT/'job.log').open('wb') as f:p=subprocess.Popen([str(ROOT.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_sit_head_control.py'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
report={'pid':p.pid,'hardwareAccess':False,'autoDeployment':False,'gpu':0,'trainingSeed':61723,'stages':'Control/Repair each 64x5 smoke then fresh seed 1024x20; cpu paired 8 normal/8 stressed trials each','sourceHashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(OUT/'launch-v3.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
