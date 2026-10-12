import math
import torch
from mjlab_microduck.tasks.microduck_roulade_mechanical_env_cfg import (
    make_roll_transfer_cfg,get_roll_spec,completion_time_cost,RollTransferRlCfg)
from mjlab_microduck.tasks.getup_mechanical_contract import LIMITS

def test_mechanical_and_target_bounds():
    spec=get_roll_spec(); model=spec.compile()
    for name,limits in LIMITS.items():
        assert max(abs(model.joint(name).range-limits))<1e-12
    cfg=make_roll_transfer_cfg()
    assert cfg.actions['joint_pos'].alpha==0
    assert cfg.actions['joint_pos'].clip==LIMITS
    assert cfg.actions['joint_pos'].scale==1

def test_single_factor_and_guards():
    control=make_roll_transfer_cfg(repair=False); repair=make_roll_transfer_cfg(repair=True)
    assert control.rewards['completion_time'].weight==0
    assert repair.rewards['completion_time'].weight==-.5
    for name in control.rewards:
        if name!='completion_time':
            assert control.rewards[name].weight==repair.rewards[name].weight
            assert control.rewards[name].params==repair.rewards[name].params
    assert repair.rewards['gentle_landing'].weight>0
    assert repair.rewards['roulade_overspeed'].weight<0
    assert repair.rewards['action_rate_l2'].weight<0
    assert RollTransferRlCfg.algorithm.learning_rate==1e-5
    assert RollTransferRlCfg.algorithm.schedule=='fixed'

def test_recovery_and_side_roll_not_completion():
    # Merely upright or no head latch does not finish the trick.
    cost=completion_time_cost(torch.tensor([0,6.,6.,6.]),torch.tensor([True,False,True,True]),
        torch.tensor([-1.,-1.,-.5,-1.]),torch.tensor([.12,.12,.12,.12]))
    assert torch.equal(cost,torch.tensor([1.,1.,1.,0.]))

def test_observation_contract():
    cfg=make_roll_transfer_cfg()
    assert cfg.observations['actor'].terms['joint_vel'].delay_min_lag==1
    assert cfg.observations['actor'].terms['joint_vel'].delay_max_lag==1
    assert cfg.observations['actor'].terms['base_ang_vel'].delay_max_lag==0
    assert cfg.commands['twist'].ranges.lin_vel_x==(0.,0.)
    assert cfg.scene.entities['robot'].articulation.actuators[0].kp_fw==5.
    assert cfg.sim.contact_sensor_maxmatch==256
