import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import mdp


def test_partial_swing_has_incremental_reward_and_idle_has_none():
    score=mdp.hd1910_stride_component_quality(torch.tensor([.02,.04,.04,.04]),torch.zeros(4),
        torch.tensor([.006,.006,.012,.012]),torch.tensor([.06,.06,.12,.12]),
        torch.tensor([.1,.1,.1,0.]))
    assert 0<score[0]<score[1]<score[2] and score[2]==1 and score[3]==0


def test_repair_does_not_suppress_instantaneous_gait_sway():
    cfg=load_env_cfg('Mjlab-WalkRepair-Flat-MicroDuck-HD1910')
    assert all(not term.params.get('straight_quality',False) for term in cfg.rewards.values())
    assert cfg.rewards['yaw_bias_cost'].params['tau']==.5
    assert cfg.rewards['yaw_error_cost'].weight==-.3
    stride=cfg.rewards['stride_quality']
    assert stride.params['base_params']['component_shaping']
    assert not stride.params['base_params']['complete_swing']
    assert not stride.params['motion_quality']
    assert cfg.commands['twist'].exact_speed==.1 and cfg.actions['joint_pos'].alpha==.45
