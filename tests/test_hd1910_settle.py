from types import SimpleNamespace
import torch
from mjlab.tasks.registry import load_env_cfg
import mjlab_microduck.tasks
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_hd1910_recovery_env_cfg import get_recovery_spec
from mjlab_microduck.tasks.microduck_hd1910_stand_env_cfg import STAND_HEIGHT

TASK="Mjlab-PushSettle-Flat-MicroDuck-HD1910"


def test_quiet_reward_requires_head_up_and_stillness_but_recovery_is_free():
    model=get_recovery_spec().compile()
    names=[model.actuator(i).name for i in range(model.nu)]
    data=SimpleNamespace(joint_pos=torch.zeros(3,14),default_joint_pos=torch.zeros(3,14),joint_vel=torch.zeros(3,14),
        root_link_pos_w=torch.tensor([[0.,0.,STAND_HEIGHT]]*3),projected_gravity_b=torch.tensor([[0.,0.,-1.]]*3),
        root_link_lin_vel_w=torch.zeros(3,3),root_link_ang_vel_b=torch.zeros(3,3))
    data.joint_pos[1,names.index("head_pitch")]=1.2
    data.root_link_ang_vel_b[2,2]=2.4
    robot=SimpleNamespace(data=data,joint_names=names,find_joints=lambda _: (list(range(14)),names))
    class Scene(dict):
        env_origins=torch.zeros(3,3)
    env=SimpleNamespace(scene=Scene(robot=robot),device="cpu",num_envs=3,step_dt=.02,common_step_counter=1000,
        action_manager=SimpleNamespace(action=torch.zeros(3,14),prev_action=torch.zeros(3,14)),
        command_manager=SimpleNamespace(get_command=lambda _:torch.zeros(3,4)))
    score=mdp.hd1910_settle_balance(env,STAND_HEIGHT)
    assert score[0]==1 and score[1]<1e-5 and score[2]<1e-5
    assert mdp.hd1910_settle_head_cost(env)[1]>0
    # Even when feedback is near HOME, a target beyond the mechanical stop
    # must receive a stronger penalty and a gradient back toward HOME.
    env.action_manager.action.requires_grad_(True)
    with torch.no_grad():
        env.action_manager.action[1,names.index("head_roll")]=4.
    target_cost=mdp.hd1910_settle_head_target_cost(env)
    assert target_cost[0]==0 and target_cost[1]>50
    target_cost.sum().backward()
    assert env.action_manager.action.grad[1,names.index("head_roll")]>0
    assert mdp.hd1910_settle_motion_cost(env)[2]>5
    env._hd1910_last_push[:]=1000
    torch.testing.assert_close(mdp.hd1910_settle_balance(env,STAND_HEIGHT),torch.ones(3))
    assert (mdp.hd1910_settle_motion_cost(env)==0).all()


def test_settle_cfg_preserves_bursts_and_interface():
    cfg=load_env_cfg(TASK)
    assert len(cfg.commands["head_pose"].ranges)==4 and len(cfg.commands["body_pose"].ranges)==6
    assert cfg.events["push_robot"].params["calm_interval"]==(4.,7.)
    assert cfg.events["push_robot"].params["stages"][-1][1]==.6
    assert cfg.rewards["quiet_motion_cost"].weight<0 and cfg.rewards["head_home_cost"].weight<0
    assert "push_robot" not in load_env_cfg(TASK,play=True).events


def test_idle_episode_schedule_gives_extended_undisturbed_experience():
    torch.manual_seed(1)
    env=SimpleNamespace(num_envs=1000,device="cpu",step_dt=.02,common_step_counter=0)
    mdp.hd1910_reset_settle_schedule(env,torch.arange(1000))
    count=(env._hd1910_next_push>1000000).sum().item()
    assert 250<count<350
