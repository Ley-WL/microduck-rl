import math
from types import SimpleNamespace
import numpy as np
import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.getup_mechanical_contract import *
from mjlab_microduck.tasks.microduck_hd1910_getup_mechanical_env_cfg import MechanicalGetupAction

TASK='Mjlab-GetupMechanical-Flat-MicroDuck-HD1910'

def test_physical_and_target_bounds_match_current_board():
    cfg=load_env_cfg(TASK)
    robot=cfg.scene.entities['robot']
    spec=robot.spec_fn()
    for collision in robot.collisions: collision.edit_spec(spec)
    model=spec.compile()
    assert cfg.actions['joint_pos'].clip==LIMITS
    assert cfg.actions['joint_pos'].alpha==.45
    for name,bounds in LIMITS.items():
        np.testing.assert_allclose(model.jnt_range[model.joint(name).id],bounds,atol=1e-12)
        assert model.jnt_limited[model.joint(name).id]
    assert cfg.scene.entities['robot'].articulation.actuators[0].kp_fw==5
    assert len([g for g in model.geom_contype if g])>0

def test_clipped_quantized_chain_retains_raw_history_and_preclip_ema():
    cfg=load_env_cfg(TASK).actions['joint_pos']
    term=MechanicalGetupAction.__new__(MechanicalGetupAction)
    term.cfg=cfg
    term._raw_actions=torch.zeros(2,14)
    term._filtered_actions=torch.zeros(2,14,dtype=torch.float64)
    term._processed_actions=torch.zeros(2,14)
    term._scale=1.
    term._offset=torch.tensor([0.,-.0873,-.4579,-.0049,.453,.0,.3491,0.,0.,0.,.0873,.4579,.0049,-.453])
    term._home64=term._offset.double()
    term._bounds64=torch.tensor(list(LIMITS.values()),dtype=torch.float64)
    term._clip=torch.tensor(list(LIMITS.values())).repeat(2,1,1)
    term._references=torch.tensor(REFERENCES,dtype=torch.float64)
    term._directions=torch.tensor(DIRECTIONS,dtype=torch.float64)
    filtered=np.zeros((2,14))
    for phase in range(20):
        raw=torch.tensor(np.random.default_rng(7001+phase).normal(0,3,(2,14)),dtype=torch.float32)
        filtered=.45*filtered+.55*raw.numpy().astype(np.float64)
        expected=encode_target(filtered+term._offset.numpy())
        term.process_actions(raw)
        np.testing.assert_allclose(term.position_target.numpy(),expected,atol=3e-7,rtol=0)
        torch.testing.assert_close(term._raw_actions,raw,atol=0,rtol=0)
    term.reset(torch.tensor([0]))
    assert torch.equal(term._raw_actions[0],torch.zeros(14))
    assert torch.equal(term._filtered_actions[0],torch.zeros(14))
    assert bool((term._filtered_actions[1]!=0).any())

def test_neck_extreme_is_bounded_without_wrapping():
    high=encode_target(np.full(14,20.))
    low=encode_target(np.full(14,-20.))
    assert abs(math.degrees(high[5])-69.9609375)<1e-9
    assert abs(math.degrees(low[5])+90)<1e-9
    assert np.max(np.abs(high))<math.pi

def test_historical_task_is_unchanged():
    cfg=load_env_cfg('Mjlab-Getup-Flat-MicroDuck-HD1910')
    assert cfg.actions['joint_pos'].clip is None
    assert cfg.actions['joint_pos'].alpha==.45
