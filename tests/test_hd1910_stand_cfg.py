import json
from pathlib import Path
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.microduck_hd1910_stand_env_cfg import PARAMS, STAND_HEIGHT, get_replica_spec, make_hd1910_stand_env_cfg

def test_replica_joint_contract_and_height():
    m = get_replica_spec().compile()
    assert m.nu == 14
    assert [m.actuator(i).name for i in range(m.nu)] == ["left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle","neck_pitch","head_pitch","head_yaw","head_roll","right_hip_yaw","right_hip_roll","right_hip_pitch","right_knee","right_ankle"]
    assert 0.1 < STAND_HEIGHT < 0.14

def test_stand_task_preserves_command_slots_and_zero_idle():
    c = load_env_cfg("Mjlab-StandHold-Flat-MicroDuck-HD1910")
    assert len(c.commands["head_pose"].ranges) == 4
    assert len(c.commands["body_pose"].ranges) == 6
    assert c.commands["twist"].rel_standing_envs == 0.9
    assert "standing_envs" not in c.curriculum
    assert all(n not in c.rewards for n in ("air_time","foot_clearance","foot_swing_height"))
    assert c.rewards["standing_composite"].weight > 0
    assert c.rewards["action_rate_l2"].weight < 0
    assert c.rewards["leg_alignment_cost"].weight < 0

def test_supplied_model_and_no_mutation_of_xl330_task():
    p = json.loads(PARAMS.read_text())
    assert p["actuator"] == "sts3215" and p["model"] == "m6"
    c = make_hd1910_stand_env_cfg()
    a = c.scene.entities["robot"].articulation.actuators[0]
    assert a.json_path == str(PARAMS) and a.motor_name is None and a.vin_min == 4.0
    c.commands["head_pose"].ranges = ()
    assert len(make_hd1910_stand_env_cfg().commands["head_pose"].ranges) == 4
    assert load_env_cfg("Mjlab-Velocity-Flat-MicroDuck").scene.entities["robot"].articulation.actuators[0].motor_name == "xl330"


def test_stand_reward_rejects_twisted_legs_but_not_head_motion():
    import torch
    from types import SimpleNamespace
    from mjlab_microduck.tasks import mdp
    names = [get_replica_spec().compile().actuator(i).name for i in range(14)]
    data = SimpleNamespace(joint_pos=torch.zeros(3,14), default_joint_pos=torch.zeros(3,14),
        root_link_pos_w=torch.tensor([[0.,0.,STAND_HEIGHT]]*3),
        projected_gravity_b=torch.tensor([[0.,0.,-1.]]*3), root_link_lin_vel_w=torch.zeros(3,3))
    # Identical balance, but one world has v1's yaw/roll/knee deformation.
    data.joint_pos[1, [0,9,10,12,13]] = torch.tensor([-.30,.45,.30,-.37,-.39])
    data.joint_pos[2, 5:9] = .3
    robot = SimpleNamespace(data=data,joint_names=names,find_joints=lambda _: (list(range(14)),names))
    class Scene(dict):
        env_origins = torch.zeros(3,3)
    env = SimpleNamespace(scene=Scene(robot=robot))
    scores = mdp.hd1910_standing_composite(env, STAND_HEIGHT)
    assert scores[0] == scores[2] == 1
    assert scores[1] < .02
    costs = mdp.hd1910_leg_pose_cost(env)
    assert costs[0] == costs[2] == 0 and costs[1] > 1
