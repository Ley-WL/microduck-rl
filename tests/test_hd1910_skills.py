import json
import numpy as np
import pytest

import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.microduck_hd1910_skills_env_cfg import PARAMS, POSE_NAMES, SIT_Z, STAND_Z


@pytest.mark.parametrize("task", ["Getup", "SitStand"])
def test_skills_use_replica_hd1910_and_runtime_contract(task):
    cfg = load_env_cfg(f"Mjlab-{task}-Flat-MicroDuck-HD1910")
    robot_cfg = cfg.scene.entities["robot"]
    spec = robot_cfg.spec_fn()
    for collision in robot_cfg.collisions:
        collision.edit_spec(spec)
    model = spec.compile()
    assert model.nu == 14 and model.nq == 21
    limits = json.loads((PARAMS.parent.parent / "xgoduck_joint_limits.json").read_text())["joints"]
    for name, entry in limits.items():
        np.testing.assert_allclose(model.jnt_range[model.joint(name).id], entry["range"], atol=1e-12, rtol=0)
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    assert actuator.json_path == str(PARAMS) and actuator.kp_fw == 5 and actuator.motor_name is None
    assert cfg.actions["joint_pos"].alpha == .45
    assert cfg.actions["joint_pos"].clip is None
    assert len(cfg.commands["head_pose"].ranges) == 4
    if "body_pose" in cfg.commands:
        assert len(cfg.commands["body_pose"].ranges) == 6
    else:
        assert cfg.observations["actor"].terms["body_command"].params["dim"] == 6
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations
    for curriculum in cfg.curriculum.values():
        if "reward_name" in curriculum.params:
            assert curriculum.params["reward_name"] in cfg.rewards
    contacts = [g for g in range(model.ngeom) if model.geom(g).name.startswith("body_ground_")]
    assert len(contacts) >= 60
    assert all(model.geom_contype[g] == 1 and model.geom_conaffinity[g] == 0 for g in contacts)
    assert all(model.geom_condim[g] == 3 for g in contacts)
    for name in ('left_foot_collision', 'right_foot_collision'):
        geom = model.geom(name).id
        assert model.geom_contype[geom] == 1 and model.geom_conaffinity[geom] == 0
        assert model.geom_condim[geom] == 3 and model.geom_priority[geom] == 1


def test_recovery_keeps_all_fallen_cohorts_in_final_curriculum():
    cfg = load_env_cfg("Mjlab-Getup-Flat-MicroDuck-HD1910")
    stages = cfg.curriculum["recovery_pose_mix"].params["param_stages"]
    assert len(POSE_NAMES) == 8
    for stage in stages:
        probabilities = stage["params"]["pose_probabilities"]
        assert sum(probabilities) == pytest.approx(1)
        assert all(p > 0 for p in probabilities)
    assert sum(stages[-1]["params"]["pose_probabilities"][2:]) == pytest.approx(.95)


def test_sitstand_height_rewards_and_command_agree_with_measured_geometry():
    cfg = load_env_cfg("Mjlab-SitStand-Flat-MicroDuck-HD1910")
    command = cfg.commands["twist"]
    assert command.sit_z == SIT_Z and command.stand_z == STAND_Z
    assert SIT_Z < .075 < STAND_Z
    for reward in cfg.rewards.values():
        if "sit_z" in reward.params:
            assert reward.params["sit_z"] == SIT_Z
        if "stand_z" in reward.params:
            assert reward.params["stand_z"] == STAND_Z
    for name in ("descent_speed", "rise_speed", "gentle_motion"):
        assert cfg.rewards[name].weight >= 0  # These functions return negative penalties.


@pytest.mark.parametrize("cohort", range(8))
def test_partial_pose_reset_keeps_other_worlds_and_terrain_offsets(cohort):
    from types import SimpleNamespace
    import torch
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.microduck_hd1910_skills_env_cfg import POSE_ANGLES, SPAWN_HEIGHTS, SITTING_TARGET_OVERRIDES
    model = load_env_cfg("Mjlab-Getup-Flat-MicroDuck-HD1910").scene.entities["robot"].spec_fn().compile()
    names = [model.actuator(i).name for i in range(14)]
    limits = torch.tensor(np.stack([model.jnt_range[model.joint(name).id] for name in names]), dtype=torch.float32).repeat(3, 1, 1)
    robot = SimpleNamespace(data=SimpleNamespace(default_joint_pos=torch.zeros(3, 14), joint_pos_limits=limits),
                            find_joints=lambda _: (list(range(14)), names))
    class Scene(dict):
        env_origins = torch.tensor([[0., 0., 0.], [0., 0., .3], [0., 0., 0.]])
    scene = Scene(robot=robot)
    scene.terrain = SimpleNamespace(env_origins=scene.env_origins)
    qpos = torch.ones(3, 21)
    qvel = torch.ones(3, 20)
    env = SimpleNamespace(device="cpu", num_envs=3, scene=scene, sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos, qvel=qvel)))
    probabilities = tuple(float(i == cohort) for i in range(8))
    mdp.hd1910_reset_skill_pose(env, torch.tensor([1]), probabilities, POSE_ANGLES, SPAWN_HEIGHTS, SITTING_TARGET_OVERRIDES,
                              joint_noise=0., tilt_noise_deg=0., clearance=.003)
    assert env._hd1910_spawn_cohort[1] == cohort
    assert qpos[1, 2] == pytest.approx(SPAWN_HEIGHTS[cohort]+.303)
    assert qpos[1, 3:7].norm() == pytest.approx(1.)
    assert torch.all(qpos[[0, 2]] == 1) and torch.all(qvel[[0, 2]] == 1)
    assert torch.all(qvel[1] == 0)
    assert torch.all(qpos[1, 7:] >= limits[1, :, 0]) and torch.all(qpos[1, 7:] <= limits[1, :, 1])
