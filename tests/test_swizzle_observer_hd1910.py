from dataclasses import asdict
from mjlab_microduck.tasks.swizzle_observer_hd1910 import make_cfg, rl_cfg
from mjlab_microduck.tasks.microduck_velocity_swizzle_env_cfg import make_microduck_velocity_swizzle_env_cfg, MicroduckSwizzleRlCfg
from mjlab_microduck.tasks.getup_mechanical_contract import LIMITS
from mjlab_microduck.robot.hardware_profile import HD1910_PARAMS
from pathlib import Path


def test_preserves_recipe():
    a, b = make_cfg(), make_microduck_velocity_swizzle_env_cfg()
    for key in ('rewards','curriculum','commands','terminations','events','observations'):
        assert repr(getattr(a,key)) == repr(getattr(b,key)),key
    for key in ('actor','critic','algorithm','num_steps_per_env','max_iterations','save_interval'):
        assert asdict(rl_cfg())[key] == asdict(MicroduckSwizzleRlCfg)[key]


def test_direction_and_hardware():
    cfg=make_cfg();spec=cfg.scene.entities['robot'].spec_fn()
    assert cfg.commands['twist'].ranges.lin_vel_x == (-0.6,0.6)
    assert cfg.commands['twist'].ranges.ang_vel_z == (-0.5,0.5)
    assert cfg.rewards['wheel_speed'].params['bidirectional'] is True
    assert 'braking' not in cfg.rewards
    assert cfg.actions['joint_pos'].clip == LIMITS
    assert {j.name:tuple(j.range) for j in spec.joints if j.name in LIMITS} == LIMITS
    wheels=[j for j in spec.joints if j.name.startswith('passive_') and 'wheel' in j.name]
    assert len(wheels)==4
    import mujoco
    model=spec.compile()
    assert all(model.jnt_limited[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,j.name)]==0 for j in wheels)
    for actuator in cfg.scene.entities['robot'].articulation.actuators:
        assert Path(actuator.json_path).resolve() == HD1910_PARAMS.resolve()
        assert actuator.kp_fw == 5.
