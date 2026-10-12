import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.xgoduck_runtime_actions import XgoDuckPositionActionCfg


def test_natural_walk_matches_runtime_and_removes_low_head_payout():
    for play in (False, True):
        cfg = load_env_cfg('Mjlab-WalkNatural-Flat-MicroDuck-HD1910', play=play)
        assert isinstance(cfg.actions['joint_pos'], XgoDuckPositionActionCfg)
        assert cfg.actions['joint_pos'].alpha == .45
        assert cfg.actions['joint_pos'].clip
        for term in cfg.rewards.values():
            if term.func == mdp.hd1910_walk_supported_reward:
                assert term.params['head_quality']
                assert term.params['head_floor'] == 0.
        error = torch.tensor([[0.,0.,0.,0.], [1.2,0.,0.,0.]])
        score = mdp.hd1910_walk_head_quality(error, std=.25, floor=0.)
        assert score[0] == 1 and score[1] < .01


def test_movement_payout_requires_commanded_direction_and_caps_overspeed():
    command=torch.tensor([[.2,0.,0.]]*5+[[0.,0.,.6],[0.,0.,0.]])
    velocity=torch.tensor([[0.,0.],[.1,0.],[.2,0.],[.4,0.],[-.2,0.],[0.,0.],[0.,0.]])
    score=mdp.hd1910_walk_motion_quality(command,velocity,
        torch.tensor([0.,0.,0.,0.,0.,.3,0.]))
    assert torch.allclose(score,torch.tensor([0.,.5,1.,1.,0.,.5,1.]))
    cfg=load_env_cfg('Mjlab-WalkNatural-Flat-MicroDuck-HD1910')
    assert cfg.rewards['track_linear_velocity'].weight==10.
    assert cfg.rewards['command_progress'].weight==8.
    assert cfg.rewards['upright'].params['motion_quality']
    assert cfg.rewards['stride_quality'].params['base_params']['require_alternation']
