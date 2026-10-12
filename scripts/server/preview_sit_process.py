"""Screen checkpoints without interrupting the isolated finite training job."""
import argparse,json
import finalize_sit_transfer as f
f.task='Mjlab-SitStandProcess-Flat-MicroDuck-HD1910'
f.experiment='hd1910_sitstand_process_v2';f.run_suffix='process1000'
f.extra_conditions={'gain-low':['--actuator-kp-scale','.8'],'gain-high':['--actuator-kp-scale','1.2']}
p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,choices=[100,500],required=True);a=p.parse_args()
logs=list((f.root/'logs/rsl_rl'/f.experiment).glob('*'+f.run_suffix));assert len(logs)==1
checkpoint=logs[0]/f'model_{a.iteration}.pt';assert checkpoint.is_file()
policy=f.evidence/f'candidate-{a.iteration}.onnx'
if not policy.exists():
    f.run(['scripts/export.py',f.task,'--checkpoint-file',checkpoint,'--onnx-file',policy,
           '--num-envs','1','--device','cuda:0'],f.evidence/f'export-{a.iteration}.log')
    f.run(['scripts/server/write_policy_metadata.py',policy,f.task],f.evidence/f'metadata-{a.iteration}.log')
reports=f.screen(policy,str(a.iteration))
summary={key:f.summarize(value)[1] for key,value in reports.items()}
(f.evidence/f'preview-{a.iteration}.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
