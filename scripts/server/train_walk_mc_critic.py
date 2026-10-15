"""Fit only the value network to full sampled returns, with whole-world holdout."""
import hashlib,json,importlib
from pathlib import Path
from dataclasses import asdict
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence'
spec_path=ROOT/'inputs/critic-spec.json'
spec=json.loads(spec_path.read_text()) if spec_path.exists() else None
module=importlib.import_module('mjlab_microduck.tasks.'+(spec['task'] if spec else 'walk_motion_critic_context'))
make_cfg,rl_cfg=module.make_cfg,module.rl_cfg
torch.manual_seed(spec['fitSeed'] if spec else 239131)
cfg=make_cfg();cfg.scene.num_envs=64
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    agent=asdict(rl_cfg());agent['upload_model']=False
    runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),agent,log_dir=None,device=env.device)
    checkpoint=ROOT/'inputs/context.pt'
    runner.load(str(checkpoint),load_cfg={'actor':True,'critic':True},map_location=env.device)
    actor,critic=runner.alg.actor.eval(),runner.alg.critic.eval()
    before={k:v.detach().clone() for k,v in actor.state_dict().items()}
    norms={k:v.detach().clone() for k,v in critic.state_dict().items() if 'normalizer' in k}
    for p in actor.parameters():p.requires_grad_(False)
    train_x=[];train_y=[];valid_x=[];valid_y=[];sources={}
    for seed in (spec['collectionSeeds'] if spec else (239101,239107)):
        path=OUT/f'critic-data-{seed}.pt';data=torch.load(path,weights_only=False)
        assert data['checkpointSha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        x,y,live,hold=data['observations'],data['returns'],data['live'],data['validationWorlds']
        worlds=len(spec['collectionCases'])*4 if spec else 48
        assert x.shape==(1500,worlds,86) and hold.sum()==worlds//4
        assert torch.equal(hold,torch.arange(worlds)%4==3)
        for selected,xx,yy in [(~hold,train_x,train_y),(hold,valid_x,valid_y)]:
            mask=live[:,selected];xx.append(x[:,selected][mask]);yy.append(y[:,selected][mask])
        sources[str(seed)]=hashlib.sha256(path.read_bytes()).hexdigest()
    x,y,vx,vy=[torch.cat(z).to(env.device) for z in (train_x,train_y,valid_x,valid_y)]
    assert all(torch.isfinite(z).all() for z in (x,y,vx,vy))
    def predict(z):return critic(TensorDict({'critic':z},batch_size=[len(z)])).squeeze(-1)
    def score():
        with torch.no_grad():
            residual=torch.cat([predict(z) for z in vx.split(4096)])-vy
            return dict(rmse=float(residual.square().mean().sqrt()),explainedVariance=float(1-residual.var(unbiased=False)/vy.var(unbiased=False)))
    optimizer=torch.optim.Adam(critic.parameters(),lr=1e-4)
    initial=score();records=[];previous=0;best=initial['rmse']
    for total in (5,100,500):
        for _ in range(previous,total):
            ids=torch.randint(len(x),(2048,),device=env.device)
            loss=(predict(x[ids])-y[ids]).square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);optimizer.step()
        assert all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
        assert all(torch.equal(v,critic.state_dict()[k]) for k,v in norms.items())
        assert all(torch.isfinite(v).all() for v in critic.state_dict().values())
        result=score();stage=OUT/f'mc-step{total}';stage.mkdir()
        runner.save(str(stage/'checkpoint.pt'));runner.export_policy_to_onnx(str(stage),'policy.onnx')
        assert hashlib.sha256((stage/'policy.onnx').read_bytes()).hexdigest()==hashlib.sha256((ROOT/'inputs/student.onnx').read_bytes()).hexdigest()
        torch.save(optimizer.state_dict(),stage/'supervised-optimizer.pt')
        records.append(dict(steps=total,**result))
        if total==5 or result['rmse']<best:
            best=min(best,result['rmse']);(OUT/'mc-selected.txt').write_text(str(stage/'checkpoint.pt'))
        (OUT/'mc-fit.json').write_text(json.dumps(dict(initial=initial,records=records,sourceHashes=sources,actorUnchanged=True,normalizersUnchanged=True,wholeWorldHoldout=True,gaitAccepted=False),indent=2))
        if total>=100 and result['rmse']>=initial['rmse']:break
        previous=total
finally:env.close()
