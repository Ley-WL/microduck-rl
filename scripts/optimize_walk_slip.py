"""Bounded policy output-bias search on slip scenarios; no hardware access."""
import concurrent.futures, hashlib, json, multiprocessing, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/walk-slip-repair-20261010'
NS=None

def init():
    global NS
    script=ROOT/'scripts/preview_xgoduck.py'
    source=script.read_text(encoding='utf-8').split('\nreset()\n')[0]
    source=source.replace('a=p.parse_args()','a=p.parse_args(["--headless"])')
    NS={'__file__':str(script),'__name__':'slip_probe'}
    exec(compile(source,str(script),'exec'),NS)
    NS['a'].headless=False
    NS['base_policy']=NS['policy']
    NS['feet']=[NS['model'].geom(n).id for n in ('left_foot_collision','right_foot_collision')]

class BiasPolicy:
    def __init__(self,base,bias):self.base=base;self.bias=bias
    def get_inputs(self):return self.base.get_inputs()
    def run(self,*args):return [self.base.run(*args)[0]+self.bias]

def bias_vector(p):
    b=np.zeros(14,np.float32)
    # Mirrored sagittal legs plus independent neck/head deltas, all radians.
    b[[2,3,4]]=p[:3];b[[11,12,13]]=-np.array(p[:3]);b[5:7]=p[3:5]
    return b

def scenario(mu=1.,speed=.4,push=0.,drop=False,seed=11,seconds=8.):
    return dict(mu=mu,speed=speed,push=push,drop=drop,seed=seed,seconds=seconds)

TRAIN=[scenario(),scenario(speed=0),scenario(mu=.2),scenario(mu=.15,speed=.2),scenario(mu=.2,push=.5,drop=True),scenario(mu=.2,push=-.5,drop=True)]
HOLD=[scenario(mu=mu,speed=v,push=p,drop=d,seed=seed,seconds=12.) for seed in (29,47) for mu,v,p,d in [(1.,0.,0.,False),(1.,.2,0.,False),(1.,.4,0.,False),(.4,.4,0.,False),(.25,.4,.4,False),(.15,.4,-.4,False),(.2,.2,0.,True),(.1,.2,0.,False)]]

def evaluate(job):
    params,cases=job
    n=NS;m=n['model'];d=n['data'];mj=n['mujoco'];trunk=m.body('trunk_base').id
    n['policy']=BiasPolicy(n['base_policy'],bias_vector(params));results=[]
    for c in cases:
        n['reset']();rng=np.random.default_rng(c['seed'])
        # Physical small initial joint variation; identical seeds for pairing.
        d.qpos[n['qadr']]+=rng.normal(0,.002,14);mj.mj_forward(m,d)
        m.geom_friction[n['feet'],0]=1. if c['drop'] else c['mu']
        max_tilt=0.;firstfall=None;xs=[];ys=[];speeds=[];action_delta=[];prev=np.zeros(14);start=None
        for tick in range(round(c['seconds']/.02)):
            t=tick*.02;n['velocity_command'][0]=c['speed'] if t>=2 else 0
            if tick==125:
                m.geom_friction[n['feet'],0]=c['mu']
                d.qvel[4]+=c['push'] # physical angular velocity impulse about body pitch
            n['step']();r=d.xmat[trunk].reshape(3,3)
            tilt=float(np.degrees(np.arccos(np.clip(r[2,2],-1,1))));max_tilt=max(tilt,max_tilt)
            if start is None and t>=2:start=d.qpos[:2].copy()
            if t>=2:
                xs.append(float(d.qpos[0]-start[0]));ys.append(float(d.qpos[1]-start[1]));speeds.append(float((r.T@d.qvel[:3])[0]))
                action_delta.append(float(np.mean((n['filtered']-prev)**2)))
            prev=n['filtered'].copy()
            if tilt>60 or not np.isfinite(d.qpos).all():firstfall=t+.02;break
        duration=(firstfall or c['seconds'])-2
        distance=xs[-1] if xs else 0.;lateral=ys[-1] if ys else 0.
        progress=distance/max(.1,c['speed']*(c['seconds']-2)) if c['speed'] else 0.
        success=firstfall is None and (distance>.04*(c['seconds']-2) if c['speed'] else abs(distance)<.06) and max_tilt<45
        score=(duration/(c['seconds']-2))*2+min(max(progress,-1),1)*3-max_tilt/90-min(abs(lateral),1)*.3
        if firstfall is not None:score-=3
        if c['speed']==0:score-=abs(distance)*10
        results.append(dict(**c,success=bool(success),firstFall=firstfall,maxTilt=max_tilt,distance=distance,lateral=lateral,meanForwardSpeed=float(np.mean(speeds)) if speeds else 0.,actionDeltaMSE=float(np.mean(action_delta)) if action_delta else 0.,score=score))
    return {'params':list(params),'score':float(np.mean([x['score'] for x in results])),'successes':sum(x['success'] for x in results),'cases':results}

def main():
    OUT.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(20261010)
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,initializer=init) as pool:
        base=pool.submit(evaluate,([0]*5,TRAIN)).result();(OUT/'baseline-train.json').write_text(json.dumps(base,indent=2))
        print('BASELINE',json.dumps(base),flush=True)
        best=base;mean=np.zeros(5);std=np.array([.035,.035,.035,.04,.04]);allrows=[base]
        for generation in range(3):
            candidates=np.clip(rng.normal(mean,std,(12,5)),-.10,.10);candidates[0]=best['params']
            rows=list(pool.map(evaluate,[(p.tolist(),TRAIN) for p in candidates]));allrows.extend(rows)
            # No lost baseline-successful scenario and no stopped walking as an escape.
            eligible=[r for r in rows if all(not b['success'] or x['success'] for b,x in zip(base['cases'],r['cases']))]
            ranked=sorted(eligible,key=lambda x:x['score'],reverse=True)
            if ranked and ranked[0]['score']>best['score']:best=ranked[0]
            elite=ranked[:4]
            if elite:mean=np.mean([r['params'] for r in elite],axis=0)
            std*=.65
            (OUT/'search.json').write_text(json.dumps(allrows,indent=2))
            print('GENERATION',generation,'BEST',best['score'],best['successes'],best['params'],flush=True)
        paired=list(pool.map(evaluate,[([0]*5,HOLD),(best['params'],HOLD)]))
    baseline,candidate=paired
    accepted=best['params']!=base['params'] and candidate['successes']>baseline['successes'] and all(not b['success'] or c['success'] for b,c in zip(baseline['cases'],candidate['cases']))
    report={'method':'Bounded derivative-free optimization of five mirrored final-layer bias parameters; not PPO retraining','sourceSha256':hashlib.sha256((ROOT.parent/'microduck-observer/models/xgoduck_walk.onnx').read_bytes()).hexdigest(),'baseline':baseline,'candidate':candidate,'accepted':accepted,'hardwareValidated':False,'limits':'16 held-out paired simulations; physical friction/gains/sensing not identified'}
    (OUT/'heldout.json').write_text(json.dumps(report,indent=2))
    import onnx
    from onnx import numpy_helper
    source=ROOT.parent/'microduck-observer/models/xgoduck_walk.onnx';model=onnx.load(source)
    for tensor in model.graph.initializer:
        if tensor.name=='mlp.6.bias':tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).copy()+bias_vector(best['params']),tensor.name))
    onnx.checker.check_model(model);dest=OUT/'slip-bias-candidate.onnx';onnx.save(model,dest)
    meta=json.loads(source.with_suffix('.metadata.json').read_text());meta.update(policySha256=hashlib.sha256(dest.read_bytes()).hexdigest(),hardwareValidated=False,optimization='slip-bias-search',acceptedInSimulation=accepted)
    dest.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2))
    print('FINAL',json.dumps({'accepted':accepted,'baselineSuccess':baseline['successes'],'candidateSuccess':candidate['successes'],'total':len(HOLD),'params':best['params']}),flush=True)

if __name__=='__main__':main()
