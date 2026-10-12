from types import SimpleNamespace
import torch
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
import mjlab_microduck.tasks
from mjlab_microduck.tasks import mdp
from mjlab_microduck.robot.sit_contact_contract import TARGET_LIMITS


def test_transfer_preserves_physics_contract_and_covers_smoothness_within_budget():
    cfg=load_env_cfg('Mjlab-SitStandTransfer-Flat-MicroDuck-HD1910')
    old=load_env_cfg('Mjlab-SitStandContact-Flat-MicroDuck-HD1910')
    assert cfg.actions['joint_pos'].alpha==.45
    assert cfg.actions['joint_pos'].clip==TARGET_LIMITS
    assert cfg.commands['twist'].alternate
    assert cfg.rewards['target_acceleration'].weight<0
    assert cfg.rewards['action_rate_l2'].weight==-0.1 and old.rewards['action_rate_l2'].weight==-.05
    assert max(s['step'] for s in cfg.curriculum['action_rate_weight'].params['weight_stages'])<1000*24
    assert 'transfer_pitch_kick' not in load_env_cfg('Mjlab-SitStandTransfer-Flat-MicroDuck-HD1910',play=True).events
    rl=load_rl_cfg('Mjlab-SitStandTransfer-Flat-MicroDuck-HD1910')
    assert rl.max_iterations==1000 and rl.algorithm.retention_coef>0 and rl.algorithm.learning_rate==1e-5


def test_target_acceleration_cost_rejects_reversal_and_clears_reset_history():
    term=SimpleNamespace(position_target=torch.zeros(2,14))
    env=SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda _:term),episode_length_buf=torch.tensor([2,2]))
    torch.testing.assert_close(mdp.hd1910_target_acceleration_cost(env),torch.zeros(2))
    term.position_target[:,3]=.1
    first=mdp.hd1910_target_acceleration_cost(env)
    term.position_target[:,3]=.2
    steady=mdp.hd1910_target_acceleration_cost(env)
    term.position_target[:,3]=.1
    reverse=mdp.hd1910_target_acceleration_cost(env)
    assert torch.all(reverse>first) and torch.all(steady<first)
    env.episode_length_buf[:]=1
    torch.testing.assert_close(mdp.hd1910_target_acceleration_cost(env),torch.zeros(2))
