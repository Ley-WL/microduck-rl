"""One bounded state-reference diagnostic, each guarded by a 64-world five-step smoke."""
import concurrent.futures,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import torch
import finalize_sit_transfer as f

root=Path(__file__).resolve().parents[2]
python=root.parent/'microduck-rl/.venv/bin/python'
environment=os.environ.copy();environment.update(PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES='6',
    OMP_NUM_THREADS='4',WANDB_MODE='offline',MUJOCO_GL='egl',MICRODUCK_WARM_START='1')
folder=root/'evidence/upright-control';folder.mkdir(parents=True,exist_ok=True)
f.evidence=folder;f.environment=environment
f.extra_conditions={'gain-low':['--actuator-kp-scale','.8'],'gain-high':['--actuator-kp-scale','1.2']}
seed=root/'logs/rsl_rl/hd1910_sit_process_ablation/transfer_seed/seed.pt'
seed.parent.mkdir(parents=True,exist_ok=True);seed.write_bytes((root/'seed.pt').read_bytes())


def command(arguments,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream:
        subprocess.run(list(map(str,arguments)),cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)


def export(task,checkpoint,output):
    command([python,'scripts/export.py',task,'--checkpoint-file',checkpoint,'--onnx-file',output,
             '--num-envs','1','--device','cuda:0'],output.with_suffix('.export.log'))
    command([python,'scripts/server/write_policy_metadata.py',output,task],output.with_suffix('.metadata.log'))


def main():
    baseline=folder/'baseline.onnx';baseline.write_bytes((root/'weight-1p0.onnx').read_bytes())
    metadata=json.loads((root/'weight-1p0.metadata.json').read_text());metadata['task']='Mjlab-SitStandProcess-Flat-MicroDuck-HD1910'
    baseline.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2))
    futures={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        original=pool.submit(f.screen,baseline,'baseline')
        for case in ('Upright',):
            case_folder=folder/case;case_folder.mkdir(parents=True,exist_ok=True)
            task='Mjlab-SitStandProcess'+case+'-Flat-MicroDuck-HD1910'
            for mode,iterations,worlds in (('smoke',5,64),('train',100,1024)):
                usage=int(subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                assert usage<200,'GPU6 is now occupied; do not interfere'
                run=case.lower()+'-'+mode
                assert not list((root/'logs/rsl_rl/hd1910_sit_process_ablation').glob('*'+run)),'Do not repeat a case'
                (folder/'progress.json').write_text(json.dumps({'case':case,'stage':mode,'iterations':iterations}))
                command([python.parent/'train',task,'--env.scene.num-envs',worlds,'--agent.max-iterations',iterations,
                         '--agent.resume','True','--agent.load-run','transfer_seed','--agent.load-checkpoint','seed.pt',
                         '--agent.run-name',run,'--agent.logger','tensorboard'],case_folder/(mode+'.log'))
                checkpoints=list((root/'logs/rsl_rl/hd1910_sit_process_ablation').glob('*'+run+'/model_'+str(iterations-1)+'.pt'))
                assert len(checkpoints)==1
                state=torch.load(checkpoints[0],map_location='cpu',weights_only=False)['actor_state_dict']
                before=torch.load(root/'seed.pt',map_location='cpu',weights_only=False)['actor_state_dict']
                assert all(bool(torch.isfinite(v).all()) for v in state.values())
                for key in before:
                    if key.startswith('obs_normalizer.'):torch.testing.assert_close(state[key],before[key],atol=0,rtol=0)
                export(task,checkpoints[0],case_folder/(mode+'.onnx'))
                if mode=='smoke':
                    (case_folder/'smoke-check.json').write_text(json.dumps({'iterations':5,'worlds':64,'normalizerExactlyPreserved':True,'finiteParameters':True}))
            futures[case]=pool.submit(f.screen,case_folder/'train.onnx',case)
        reference=original.result();output={'hardwareValidated':False,'baseline':{k:f.summarize(v)[1] for k,v in reference.items()},'cases':{}}
        for case,future in futures.items():
            reports=future.result();stats={}
            for condition,report in reports.items():
                old,_=f.summarize(reference[condition]);current,item=f.summarize(report)
                assert set(old)==set(current)
                for key in old:np.testing.assert_allclose(old[key]['initialQpos'],current[key]['initialQpos'],atol=1e-12)
                item['regressions']=[list(key) for key,row in old.items() if row['success'] and not current[key]['success']]
                stats[condition]=item
            output['cases'][case]=stats
            (folder/'comparison.json').write_text(json.dumps(output,indent=2));print(json.dumps({'case':case,'conditions':stats}),flush=True)
    (folder/'completed.txt').write_text('complete\n')


if __name__=='__main__':
    try:main()
    except BaseException:
        (folder/'job.exit').write_text('1');raise
    else:(folder/'job.exit').write_text('0')
