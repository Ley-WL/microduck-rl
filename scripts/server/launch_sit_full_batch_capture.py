import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=root/'evidence/full-batch-failure-capture';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Follow existing launch'
screen=root/'evidence/full-batch-control';assert (screen/'job.exit').read_text()=='0'
files=['scripts/server/capture_sit_full_batch_failures.py','scripts/server/launch_sit_full_batch_capture.py','scripts/evaluate_skills_cpu.py','scripts/bam_cpu_controller.py']
files += [str(p.relative_to(root)) for p in sorted(screen.rglob('*')) if p.is_file() and (p.suffix=='.onnx' or 'screening' in str(p) and p.suffix in ('.json','.npz'))]
files += [str(p.relative_to(root)) for p in sorted(screen.rglob('*.metadata.json'))]
files += [str(p.relative_to(root)) for p in sorted((root/'src').rglob('*')) if p.is_file() and '__pycache__' not in str(p)]
manifest={'training':False,'hardwareAccess':False,'gpuAccess':False,'sourceHashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}
(folder/'launch.json').write_text(json.dumps(manifest,indent=2))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2')
with (folder/'job.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/capture_sit_full_batch_failures.py'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(folder/'launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'pid':process.pid,'training':False,'gpuAccess':False}))
