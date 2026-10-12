import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import mdp


def test_swing_feedback_improves_before_the_old_125ms_threshold():
    height=torch.tensor([[.004,0.],[.008,0.],[.01,0.],[.01,.01],[0.,0.],[.01,0.]])
    air=torch.tensor([[.06,0.],[.08,0.],[.1,0.],[.1,.1],[0.,0.],[.3,0.]])
    contact=torch.tensor([[False,True]]*3+[[False,False],[True,True],[False,True]])
    score=mdp.hd1910_continuous_swing_quality(height,air,contact)
    assert 0<score[0]<score[1]<score[2] and score[2]==1
    assert (score[3:]==0).all()


def test_swing_replaces_dead_air_signal_and_preserves_runtime_contract():
    cfg=load_env_cfg('Mjlab-WalkSwing-Flat-MicroDuck-HD1910')
    assert cfg.rewards['air_time'].params['base_func']==mdp.hd1910_walk_swing_progress
    assert cfg.rewards['air_time'].params['motion_quality']
    assert cfg.rewards['air_time'].params['head_quality']
    assert cfg.actions['joint_pos'].alpha==.45 and cfg.commands['twist'].exact_speed==.1


def test_focus_transfers_reward_mass_without_changing_physics_or_targets():
    cfg=load_env_cfg('Mjlab-WalkSwingFocus-Flat-MicroDuck-HD1910')
    base=load_env_cfg('Mjlab-WalkSwing-Flat-MicroDuck-HD1910')
    assert cfg.rewards['air_time'].weight==8.
    assert cfg.rewards['track_linear_velocity'].weight==5.
    assert cfg.rewards['command_progress'].weight==1.
    assert cfg.rewards['air_time'].params==base.rewards['air_time'].params
    assert cfg.commands['twist']==base.commands['twist']
    assert cfg.actions['joint_pos']==base.actions['joint_pos']
