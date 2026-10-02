from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
import mjlab_microduck.tasks
from mjlab_microduck.tasks.microduck_hd1910_stand_env_cfg import PARAMS

TASK='Mjlab-Walk-Flat-MicroDuck-HD1910'

def test_walk_preserves_replica_physics_and_command_contract():
    cfg=load_env_cfg(TASK)
    model=cfg.scene.entities['robot'].spec_fn().compile()
    assert model.nu==14 and sum('body_ground_' in model.geom(i).name for i in range(model.ngeom))==68
    assert cfg.scene.entities['robot'].articulation.actuators[0].json_path==str(PARAMS)
    assert cfg.commands['twist'].ranges.lin_vel_x==(-.2,.3)
    assert cfg.commands['twist'].rel_turn_in_place_envs==.15
    assert len(cfg.commands['head_pose'].ranges)==4 and len(cfg.commands['body_pose'].ranges)==6
    assert all(name in cfg.rewards for name in ('air_time','foot_clearance','foot_swing_height','idle_home'))
    assert all(name not in cfg.rewards for name in ('quiet_motion_cost','leg_alignment_cost'))
    assert cfg.rewards['head_target_cost'].weight<0
    assert load_rl_cfg(TASK).experiment_name=='microduck_hd1910_walk'

def test_walk_play_has_no_push_or_head_command_drift():
    cfg=load_env_cfg(TASK,play=True)
    assert 'push_robot' not in cfg.events
    assert all(r==(0.,0.) for r in cfg.commands['head_pose'].ranges)
