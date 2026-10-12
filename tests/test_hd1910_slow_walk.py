import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import mdp


def test_slow_walk_targets_exact_point_one_and_complete_swings():
    for play in (False,True):
        cfg=load_env_cfg('Mjlab-WalkSlow-Flat-MicroDuck-HD1910',play=play)
        twist=cfg.commands['twist']
        assert twist.exact_speed==.1 and twist.exact_speed_prob==.8
        assert twist.straight_speed_range==(.07,.13)
        assert twist.straight_forward_prob==1.
        stride=cfg.rewards['stride_quality'].params['base_params']
        assert stride['require_alternation'] and stride['complete_swing']
        assert stride['target_height']==.015 and stride['cycle_time']==.6
        assert cfg.rewards['straight_path_cost'].weight<0
        assert cfg.rewards['command_progress'].params['straight_quality']
        assert cfg.rewards['track_linear_velocity'].params['base_params']['std']==.04
        assert cfg.actions['joint_pos'].alpha==.45
