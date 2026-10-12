import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('head_start',Path(__file__).parents[1]/'scripts/server/prepare_head_start_checkpoint.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_reinitialize_only_head_outputs_and_optimizer_rows():
    actor={'mlp.6.weight':torch.randn(14,128),'mlp.6.bias':torch.randn(14),
           'distribution.std_param':torch.ones(14)*.5,'obs_normalizer._mean':torch.randn(1,61),
           'mlp.0.weight':torch.randn(512,61)}
    original={'actor_state_dict':actor,'critic_state_dict':{'weight':torch.randn(1,128)},
              'optimizer_state_dict':{'state':{0:{'exp_avg':torch.ones(14),'step':torch.tensor(99.)},
                                               1:{'exp_avg_sq':torch.ones(14,128)},
                                               2:{'exp_avg':torch.ones(512,61)}}}}
    result=module.prepare(original)
    leg=[0,1,2,3,4,9,10,11,12,13]
    for name in ('mlp.6.weight','mlp.6.bias','distribution.std_param'):
        assert torch.equal(result['actor_state_dict'][name][leg],actor[name][leg])
    assert not result['actor_state_dict']['mlp.6.weight'][5:9].any()
    assert not result['actor_state_dict']['mlp.6.bias'][5:9].any()
    assert torch.allclose(result['actor_state_dict']['distribution.std_param'][5:9],torch.ones(4)*.1)
    assert torch.equal(result['actor_state_dict']['obs_normalizer._mean'],actor['obs_normalizer._mean'])
    assert torch.equal(result['actor_state_dict']['mlp.0.weight'],actor['mlp.0.weight'])
    assert torch.equal(result['critic_state_dict']['weight'],original['critic_state_dict']['weight'])
    state=result['optimizer_state_dict']['state']
    assert not state[0]['exp_avg'][5:9].any()
    assert not state[1]['exp_avg_sq'][5:9].any()
    assert state[0]['step']==99
    assert state[2]['exp_avg'].eq(1).all()
    assert original['optimizer_state_dict']['state'][0]['exp_avg'].eq(1).all()
    partial=module.prepare(original,head_retain=.25)
    assert torch.equal(partial['actor_state_dict']['mlp.6.weight'][5:9],actor['mlp.6.weight'][5:9]*.25)


def test_head_start_keeps_real_action_contract_and_failure_cost():
    from mjlab_microduck.tasks.microduck_hd1910_walk_headstart_env_cfg import make_hd1910_walk_headstart_env_cfg
    from mjlab_microduck.tasks.microduck_hd1910_walk_env_cfg import make_hd1910_walk_env_cfg
    base=make_hd1910_walk_env_cfg();head=make_hd1910_walk_headstart_env_cfg()
    assert head.actions['joint_pos'].scale==base.actions['joint_pos'].scale
    assert head.actions['joint_pos'].clip==base.actions['joint_pos'].clip
    assert head.rewards['failure_cost'].weight==-5
    assert head.rewards['head_pose_bias'].weight==2.4
