"""Compare seed and intermediate/final exports; never deploy a regressed actor."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[2]
python=root.parent/'microduck-rl/.venv/bin/python'
task='Mjlab-SitStandTransfer-Flat-MicroDuck-HD1910'
experiment='hd1910_sitstand_transfer_v1'
run_suffix='retained-smooth1000'
extra_conditions={}
evidence=root/'evidence/train'
environment=os.environ.copy();environment['PYTHONPATH']=str(root/'src')


def run(arguments,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream:subprocess.run([str(python),*map(str,arguments)],cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)


def screen(policy,tag):
    folder=evidence/'screening'/tag
    # Reuse an intermediate read-only screen if it used exactly this model
    # and the corrected independent-trial evaluator. Never reuse old physics.
    sha=hashlib.sha256(Path(policy).read_bytes()).hexdigest()
    conditions=('nominal','stress',*extra_conditions)
    paths=[folder/condition/'summary.json' for condition in conditions]
    if all(p.is_file() for p in paths):
        saved={condition:json.loads((folder/condition/'summary.json').read_text()) for condition in conditions}
        complete=True
        for condition,report in saved.items():
            scenarios=('stand_cycle','sit_cycle','stand_hold','sit_hold') if condition=='nominal' else ('stand_cycle','sit_cycle')
            expected={(scenario,seed) for scenario in scenarios for seed in (8111,9001)}
            actual={(g['scenario'],g['seed']) for g in report['groups']}
            complete &= actual==expected and len(report['groups'])==len(expected)
            complete &= all(g['trials']==4 and len(g['results'])==4 and {r['trial'] for r in g['results']}==set(range(4)) for g in report['groups'])
        if complete and all(r['policySha256']==sha and r['configuration'].get('independentTrialFrictionReset')
                and r['configuration'].get('frictionConstraintRevision')=='dof-id-friction-force-v1' for r in saved.values()):return saved
    run(['scripts/evaluate_skills_cpu.py','--onnx',policy,'--output',folder/'nominal',
         '--trials','4','--seeds','8111','9001'],folder/'nominal.log')
    run(['scripts/evaluate_skills_cpu.py','--onnx',policy,'--output',folder/'stress',
         '--trials','4','--seeds','8111','9001','--scenario','stand_cycle','sit_cycle',
         '--transition-stress','--joint-velocity-lag-steps','1'],folder/'stress.log')
    for name,extra in extra_conditions.items():
        run(['scripts/evaluate_skills_cpu.py','--onnx',policy,'--output',folder/name,
             '--trials','4','--seeds','8111','9001','--scenario','stand_cycle','sit_cycle',*extra],folder/(name+'.log'))
    return {condition:json.loads((folder/condition/'summary.json').read_text()) for condition in conditions}


def summarize(report):
    rows={(g['scenario'],g['seed'],r['trial']):r for g in report['groups'] for r in g['results']}
    edges=[t for r in rows.values() for t in r['transitionSmoothness'] if t['time']>0]
    return rows,{'successes':sum(r['success'] for r in rows.values()),'trials':len(rows),
                'meanKneeExcessTravelDeg':np.mean([t['excessTravelDeg'] for t in edges],axis=0).tolist(),
                'p95MaxKneeFrameChangeDeg':np.percentile([t['maxFrameChangeDeg'] for t in edges],95,axis=0).tolist()}


def main():
    baseline=evidence/'baseline.onnx';shutil.copy2(root/'weight-1p0.onnx',baseline)
    metadata=json.loads((root/'weight-1p0.metadata.json').read_text())
    metadata['task']=task
    baseline.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2))
    baseline_reports=screen(baseline,'baseline')
    baseline_summary={k:summarize(v)[1] for k,v in baseline_reports.items()}
    outcome={'hardwareValidated':False,'formalAccepted':False,'baseline':baseline_summary,
             'selectionRule':'No lost successful paired trial in either condition; mean excess knee travel improves at least 5% in stress, neither knee p95 max frame jump worsens over 5%. Diagnostic tolerance, not servo protection.',
             'candidates':[],'selected':None}
    (evidence/'comparison.json').write_text(json.dumps(outcome,indent=2))
    logs=list((root/'logs/rsl_rl'/experiment).glob('*'+run_suffix))
    assert len(logs)==1
    for iteration in (100,500,999):
        checkpoint=logs[0]/f'model_{iteration}.pt'
        policy=evidence/f'candidate-{iteration}.onnx'
        if not policy.is_file():
            run(['scripts/export.py',task,'--checkpoint-file',checkpoint,'--onnx-file',policy,
                 '--num-envs','1','--device','cuda:0'],evidence/f'export-{iteration}.log')
            run(['scripts/server/write_policy_metadata.py',policy,task],evidence/f'metadata-{iteration}.log')
        reports=screen(policy,str(iteration));conditions={};lost=0
        for condition,report in reports.items():
            original,_=summarize(baseline_reports[condition]);candidate,stats=summarize(report)
            assert set(original)==set(candidate)
            for key in original:np.testing.assert_allclose(original[key]['initialQpos'],candidate[key]['initialQpos'],atol=1e-12)
            regressions=[list(key) for key,row in original.items() if row['success'] and not candidate[key]['success']]
            lost+=len(regressions)
            stats['regressions']=regressions;conditions[condition]=stats
        before=baseline_summary['stress'];after=conditions['stress']
        ratio=float(np.mean(after['meanKneeExcessTravelDeg'])/max(np.mean(before['meanKneeExcessTravelDeg']),1e-8))
        frame_ok=all(v<=b*1.05+1e-6 for v,b in zip(after['p95MaxKneeFrameChangeDeg'],before['p95MaxKneeFrameChangeDeg']))
        eligible=lost==0 and ratio<=.95 and frame_ok
        item={'iteration':iteration,'sha256':hashlib.sha256(policy.read_bytes()).hexdigest(),
              'conditions':conditions,'stressExcessTravelRatio':ratio,'eligible':eligible}
        outcome['candidates'].append(item)
        if eligible and (outcome['selected'] is None or ratio<outcome['selected']['stressExcessTravelRatio']):outcome['selected']=item
        (evidence/'comparison.json').write_text(json.dumps(outcome,indent=2));print(json.dumps(item),flush=True)
    # Final actor GPU transfer profile: additional feedback disturbance/gain
    # hypotheses plus both endpoints. This is screening, not formal acceptance.
    run(['scripts/server/evaluate_skills.py','--kind','sitstand','--onnx',evidence/'policy.onnx',
         '--output',evidence/'gpu-robust','--envs','32','--seeds','1009','2027','--profile','robust'],evidence/'gpu-robust.log')
    print(json.dumps({'selectedIteration':None if outcome['selected'] is None else outcome['selected']['iteration'],
                      'hardwareAccess':False,'deployed':False}),flush=True)


if __name__=='__main__':main()
