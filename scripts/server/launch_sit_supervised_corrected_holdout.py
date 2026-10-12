"""Fresh unseen seeds for best eligible frozen supervised actor; two CPU workers."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];screen=root/'evidence/supervised-corrected-screen'
assert (screen/'job.exit').read_text().strip()=='0'
assert json.loads((screen/'source-check.json').read_text())['sourceHashesExactlyPreserved']
results=json.loads((screen/'comparison.json').read_text())
eligible=[k for k,v in results['cases'].items() if v['eligibleForHoldout']]
assert eligible,'No eligible candidate'
candidate=min(eligible,key=lambda k:results['cases'][k]['stressExcessTravelRatio'])
folder=root/'evidence/supervised-corrected-holdout';folder.mkdir(parents=True,exist_ok=True)
assert not (folder/'launch.json').exists(),'Follow existing launch'
files=['scripts/evaluate_skills_cpu.py','scripts/bam_cpu_controller.py','scripts/server/finalize_sit_transfer.py','scripts/server/run_sit_supervised_corrected_holdout.py','evidence/supervised-corrected-screen/comparison.json']
files += ['policies/'+p+s for p in ('baseline',candidate) for s in ('.onnx','.metadata.json')]
files += [str(p.relative_to(root)) for p in sorted((root/'src').rglob('*')) if p.is_file() and '__pycache__' not in str(p)]
manifest={'hardwareAccess':False,'gpuAccess':False,'training':False,'seeds':[33013,44017],'eligibleCases':eligible,'candidate':candidate,'policySha256':results['cases'][candidate]['policySha256'],'sourceHashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}
(folder/'launch.json').write_text(json.dumps(manifest,indent=2))
with (folder/'job.log').open('wb') as log:
    process=subprocess.Popen([str(root.parent/'microduck-rl/.venv/bin/python'),'scripts/server/run_sit_supervised_corrected_holdout.py'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(folder/'launch.tmp').write_text(json.dumps(manifest,indent=2));(folder/'launch.tmp').replace(folder/'launch.json')
print(json.dumps({'pid':process.pid,'candidate':candidate,'seeds':manifest['seeds'],'eligibleCases':eligible}))
