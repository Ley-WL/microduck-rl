from types import SimpleNamespace
import torch
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_hd1910_dance_env_cfg import make_hd1910_dance_env_cfg


def test_phrase_timing_and_smooth_wrap():
    times = torch.tensor([0., 1., 1.6, 2.2, 6.4, 7., 7.99, 8., 9.])
    x = mdp.duck_dance_targets(times)
    assert x.shape == (9, 10)
    assert torch.all(x[[0, 1, 5, 6, 7, 8]].abs() < 1e-5)
    assert x[2, 2] > .24 and x[3, 2] < -.24
    assert x[4, 1] < -.15
    t = torch.linspace(0, 16, 16001)
    y = mdp.duck_dance_targets(t)
    assert torch.isfinite(y).all()
    assert (y[1:] - y[:-1]).abs().max() < .002


def test_hardware_commands_and_conflicting_rewards():
    cfg = make_hd1910_dance_env_cfg()
    assert cfg.scene.entities['robot'].articulation.actuators[0].kp_fw == 5
    assert cfg.actions['joint_pos'].alpha == .45
    assert cfg.episode_length_s == 16
    assert cfg.commands['head_pose'].part == 'head'
    assert cfg.commands['body_pose'].part == 'body'
    assert 'standing_composite' not in cfg.rewards
    assert cfg.rewards['dance_tracking'].weight > 0
    assert cfg.rewards['action_rate_l2'].weight < 0
    assert cfg.scene.entities['robot'].spec_fn().compile().nu == 14


def test_fallen_head_tracking_cannot_farm_reward(monkeypatch):
    monkeypatch.setattr(mdp, 'head_pose_tracking', lambda *a, **kw: torch.ones(3))
    class Scene(dict):
        terrain = SimpleNamespace(env_origins=torch.zeros(3, 3))
    data = SimpleNamespace(
        root_link_pos_w=torch.tensor([[0., 0., .12], [0., 0., .03], [0., 0., .12]]),
        root_link_quat_w=torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.], [0., 1., 0., 0.]]))
    env = SimpleNamespace(scene=Scene(robot=SimpleNamespace(data=data)),
        command_manager=SimpleNamespace(get_command=lambda name: torch.zeros(3, 6)))
    value = mdp.duck_dance_tracking(env, .12)
    assert value[0] == 1 and value[1] == 0 and value[2] == 0


def test_stability_recipe_keeps_dance_and_activates_zero_twist_slip():
    from mjlab_microduck.tasks.microduck_hd1910_dance_env_cfg import make_hd1910_dance_stable_env_cfg
    old, new = make_hd1910_dance_env_cfg(), make_hd1910_dance_stable_env_cfg()
    assert old.rewards['foot_slip'].params['command_threshold'] == .01
    assert new.rewards['foot_slip'].params['command_threshold'] < 0
    assert new.commands['head_pose'] == old.commands['head_pose']
    assert new.commands['body_pose'] == old.commands['body_pose']
    assert new.actions == old.actions
    assert new.terminations == old.terminations


def test_balance_cost_sign_and_dance_roll_allowance():
    angle=torch.tensor([0.,.06,.3])
    q=torch.stack([torch.cos(angle/2),torch.sin(angle/2),torch.zeros(3),torch.zeros(3)],dim=1)
    data=SimpleNamespace(root_link_quat_w=q,root_link_ang_vel_b=torch.zeros(3,3))
    env=SimpleNamespace(scene={'robot':SimpleNamespace(data=data)})
    values=mdp.duck_dance_balance_cost(env)
    assert values[0] == values[1] == 0 and values[2] > 0


def test_recovery_only_widens_training_disturbance():
    from mjlab_microduck.tasks.microduck_hd1910_dance_env_cfg import make_hd1910_dance_recovery_env_cfg, make_hd1910_dance_stable_env_cfg
    old,new=make_hd1910_dance_stable_env_cfg(),make_hd1910_dance_recovery_env_cfg()
    assert new.commands==old.commands and new.actions==old.actions
    assert new.rewards==old.rewards and new.terminations==old.terminations
    assert new.events['push_robot'].params['velocity_range']['x']==(-.16,.16)


def test_expressive_full_amplitude_limits_and_curriculum():
    import re
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME
    from mjlab_microduck.tasks.microduck_hd1910_dance_env_cfg import make_hd1910_dance_expressive_env_cfg
    cfg=make_hd1910_dance_expressive_env_cfg(play=True)
    command=object.__new__(mdp.DuckDanceCommand)
    command.cfg=cfg.commands['head_pose']
    command._env=SimpleNamespace(common_step_counter=0)
    times=torch.linspace(0,8,8001)
    targets=command._targets(times)
    assert abs(float(targets[:,2].max())-.5)<1e-5
    assert abs(float(targets[:,3].abs().max())-.15)<1e-5
    assert abs(float(targets[:,6].min())+.015)<1e-5
    model=cfg.scene.entities['robot'].spec_fn().compile()
    for i,name in enumerate(('neck_pitch','head_pitch','head_yaw','head_roll')):
        home=next(v for pattern,v in HOME_FRAME.joint_pos.items() if re.fullmatch(pattern,name))
        lo,hi=model.joint(name).range
        assert float(targets[:,i].min())+home>lo+.05
        assert float(targets[:,i].max())+home<hi-.05
    command.cfg=make_hd1910_dance_expressive_env_cfg().commands['head_pose']
    start=command._targets(times)
    command._env.common_step_counter=7200
    assert torch.allclose(command._targets(times),targets)
    assert float(start[:,2].max()) < float(targets[:,2].max())


def test_resilient_keeps_full_action_contract_and_termination():
    from mjlab_microduck.tasks.microduck_hd1910_dance_env_cfg import make_hd1910_dance_expressive_env_cfg,make_hd1910_dance_resilient_env_cfg
    old,new=make_hd1910_dance_expressive_env_cfg(),make_hd1910_dance_resilient_env_cfg()
    assert new.actions==old.actions and new.terminations==old.terminations
    assert new.rewards==old.rewards
    for key in ('head_pose','body_pose'):
        assert new.commands[key].gains==old.commands[key].gains
        assert new.commands[key].ramp_steps==0
    assert new.episode_length_s==24
    assert new.events['push_robot'].interval_range_s==(1.5,4.)
