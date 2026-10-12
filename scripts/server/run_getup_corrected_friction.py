"""Repeat getup diagnostic cohorts only because the CPU physics was repaired."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=root/'evidence/getup-corrected';folder.mkdir(parents=True,exist_ok=True)
python=root.parent/'microduck-rl/.venv/bin/python'
cases={'normal':('getup',[]),'stress':('getup',['--transition-stress','--joint-velocity-lag-steps','1']),
    'contact-normal':('getup',['--contact-repair']),
    'contact-stress':('getup',['--contact-repair','--transition-stress','--joint-velocity-lag-steps','1']),
    'seated-original':('getup',['--getup-seated-probe','--scenario','sit','stand']),
    'seated-v3':('getup-v3',['--getup-seated-probe','--scenario','sit','stand'])}

if __name__=='__main__':
    try:
        assert not (folder/'summary.json').exists(),'Preserve existing results'
        environment=os.environ.copy();environment['OMP_NUM_THREADS']='1'
        outcome={'hardwareAccess':False,'formalAccepted':False,'frictionConstraintRevision':'dof-id-friction-force-v1','cases':{}}
        for name,(policy,extra) in cases.items():
            with (folder/(name+'.log')).open('w') as log:
                subprocess.run([str(python),'scripts/evaluate_skills_cpu.py','--onnx',str(root/'policies'/f'{policy}.onnx'),
                    '--output',str(folder/name),'--trials','4','--seeds','8111','9001',*extra],cwd=root,env=environment,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads((folder/name/'summary.json').read_text());assert report['configuration']['frictionConstraintRevision']=='dof-id-friction-force-v1'
            rows=[r for g in report['groups'] for r in g['results']]
            outcome['cases'][name]={'successes':sum(r['success'] for r in rows),'trials':len(rows),'policySha256':report['policySha256']}
            (folder/'summary.json').write_text(json.dumps(outcome,indent=2));print(json.dumps({name:outcome['cases'][name]}),flush=True)
        manifest=json.loads((folder/'launch.json').read_text())
        assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==v for n,v in manifest['sourceHashes'].items())
        outcome['sourceHashesExactlyPreserved']=True;(folder/'summary.json').write_text(json.dumps(outcome,indent=2))
        (folder/'completed.txt').write_text('complete\n');(folder/'job.exit').write_text('0')
    except BaseException:
        (folder/'job.exit').write_text('1');raise
