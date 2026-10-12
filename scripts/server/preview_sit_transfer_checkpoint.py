"""Read-only intermediate screening while the finite PPO job continues."""
import argparse,json
import finalize_sit_transfer as f
p=argparse.ArgumentParser();p.add_argument('--iteration',type=int,required=True);a=p.parse_args()
logs=list((f.root/'logs/rsl_rl/hd1910_sitstand_transfer_v1').glob('*retained-smooth1000'))
assert len(logs)==1
checkpoint=logs[0]/f'model_{a.iteration}.pt';assert checkpoint.is_file()
policy=f.evidence/f'candidate-{a.iteration}.onnx'
if not policy.exists():
    f.run(['scripts/export.py',f.task,'--checkpoint-file',checkpoint,'--onnx-file',policy,
           '--num-envs','1','--device','cuda:0'],f.evidence/f'export-{a.iteration}.log')
    f.run(['scripts/server/write_policy_metadata.py',policy,f.task],f.evidence/f'metadata-{a.iteration}.log')
reports=f.screen(policy,str(a.iteration))
summary={k:f.summarize(v)[1] for k,v in reports.items()}
(f.evidence/f'preview-{a.iteration}.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
