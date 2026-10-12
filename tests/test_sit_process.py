from types import SimpleNamespace
import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab_microduck.tasks.sit_process_cost import knee_process_cost,reset_transition_state
from mjlab_microduck.tasks.sit_process_cost import randomize_process_gains


def test_cost_penalizes_knee_reversal_and_reset_not_head_motion():
    term=SimpleNamespace(_target_names=['left_knee','head_pitch','right_knee'],position_target=torch.zeros(2,3))
    env=SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda _:term),episode_length_buf=torch.tensor([2,2]))
    knee_process_cost(env)
    term.position_target[:,1]=100.
    torch.testing.assert_close(knee_process_cost(env),torch.zeros(2))
    term.position_target[:,0]=.05
    first=knee_process_cost(env)
    term.position_target[:,0]=.1
    steady=knee_process_cost(env)
    term.position_target[:,0]=.05
    reverse=knee_process_cost(env)
    assert torch.all(reverse>first) and torch.all(first>steady)
    env.episode_length_buf[:]=1
    torch.testing.assert_close(knee_process_cost(env),torch.zeros(2))


def test_process_candidate_keeps_contract_and_matches_disturbance_frequency():
    cfg=load_env_cfg('Mjlab-SitStandProcess-Flat-MicroDuck-HD1910')
    old=load_env_cfg('Mjlab-SitStandTransfer-Flat-MicroDuck-HD1910')
    assert cfg.actions['joint_pos'].alpha==old.actions['joint_pos'].alpha==.45
    assert cfg.actions['joint_pos'].clip==old.actions['joint_pos'].clip
    assert cfg.events['transfer_pitch_kick'].params['probability']==1.
    assert old.events['transfer_pitch_kick'].params['probability']==.25
    assert 'transfer_pitch_kick' not in load_env_cfg('Mjlab-SitStandProcess-Flat-MicroDuck-HD1910',play=True).events
    algo=load_rl_cfg('Mjlab-SitStandProcess-Flat-MicroDuck-HD1910').algorithm
    assert algo.temporal_coef==3. and algo.onpolicy_retention_coef==.2


def test_reset_bookkeeping_resynchronizes_after_command_reset():
    env=SimpleNamespace(num_envs=2,device='cpu',_sit_transfer_command=torch.ones(2),
                        _sit_transfer_edge_tick=torch.ones(2,dtype=torch.long),
                        command_manager=SimpleNamespace(get_command=lambda _:torch.zeros(2,3)))
    reset_transition_state(env,torch.tensor([1]))
    assert not env._process_transition_synced[1]
    assert env._sit_transfer_command[1]==0 and env._sit_transfer_edge_tick[1]==-100000


def test_gain_randomization_reaches_full_range_without_accumulation():
    from bam.mjlab import BamActuator
    actuator=BamActuator.__new__(BamActuator)
    actuator.kp_scale=torch.ones(4096,1);actuator.kd_scale=torch.ones(4096,1)
    actuator.default_kp_scale=torch.ones(4096,1);actuator.default_kd_scale=torch.ones(4096,1)
    env=SimpleNamespace(num_envs=4096,device='cpu',scene={'robot':SimpleNamespace(actuators=[actuator])})
    torch.manual_seed(3019)
    for _ in range(2):
        randomize_process_gains(env,None)
        assert .8<=actuator.kp_scale.min()<.81 and 1.19<actuator.kp_scale.max()<=1.2
        assert actuator.kp_scale.std()>.1
        assert .9<=actuator.kd_scale.min() and actuator.kd_scale.max()<=1.1
