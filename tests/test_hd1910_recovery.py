from types import SimpleNamespace
import pytest
import torch
import mujoco
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import mdp

TASK = "Mjlab-PushRecovery-Flat-MicroDuck-HD1910"


def test_recovery_cfg_keeps_interface_and_allows_dynamic_recovery():
    c=load_env_cfg(TASK)
    assert len(c.commands["head_pose"].ranges)==4 and len(c.commands["body_pose"].ranges)==6
    assert c.rewards["leg_alignment_cost"].weight<0
    assert c.rewards["standing_composite"].weight>0
    assert c.events["push_robot"].params["stages"][-1][1]==.6
    assert c.events["push_robot"].interval_range_s==(.1,.1)
    assert len(c.scene.entities["robot"].collisions)==2
    assert c.scene.terrain.collisions[0].conaffinity & 4
    assert "push_robot" not in load_env_cfg(TASK,play=True).events


def test_push_is_additive_and_grace_is_only_external():
    velocity=torch.ones(16,6)*.05
    robot=SimpleNamespace(data=SimpleNamespace(root_link_vel_w=velocity,
        root_link_quat_w=torch.tensor([[1.,0,0,0]]*16)))
    def write(v,env_ids):
        velocity[env_ids]=v
    robot.write_root_link_velocity_to_sim=write
    env=SimpleNamespace(scene={"robot":robot},num_envs=16,device="cpu",step_dt=.02,common_step_counter=100)
    ids=torch.arange(16)
    mdp._hd1910_push_state(env)
    assert (mdp.hd1910_recovery_pose_factor(env)==1).all()
    before=velocity.clone()
    mdp.hd1910_burst_push(env,ids,((0,.30),))
    delta=velocity-before
    assert (delta[:,2:]==0).all()
    magnitude=delta[:,:2].norm(dim=-1)
    assert (magnitude>=.08-1e-6).all() and (magnitude<=.30+1e-6).all()
    assert (mdp.hd1910_recovery_pose_factor(env)==.1).all()
    assert (env._hd1910_next_push>100).all()
    env.common_step_counter=201
    assert (mdp.hd1910_recovery_pose_factor(env)==1).all()
    mdp.hd1910_reset_push_schedule(env,torch.tensor([0]))
    assert env._hd1910_burst_left[0]==0 and env._hd1910_last_push[0]==-10000


@pytest.mark.skipif(not torch.cuda.is_available(),reason="Warp CUDA")
def test_compiled_contacts_and_real_rollout():
    from mjlab.envs import ManagerBasedRlEnv
    cfg=load_env_cfg(TASK);cfg.scene.num_envs=4
    env=ManagerBasedRlEnv(cfg,device="cuda:0")
    try:
        m=env.sim.mj_model
        bodies=[i for i in range(m.ngeom) if "body_ground_" in m.geom(i).name]
        assert len(bodies)==68
        assert all(m.geom_contype[i]==4 and m.geom_conaffinity[i]==0 for i in bodies)
        planes=[i for i in range(m.ngeom) if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE]
        assert planes and all(m.geom_conaffinity[i]&4 for i in planes)
        obs,_=env.reset(seed=12)
        assert obs["actor"].shape==(4,61) and env.action_manager.total_action_dim==14
        env._hd1910_next_push[:]=0
        for _ in range(30):
            obs,reward,_,_,_=env.step(torch.zeros((4,14),device=env.device))
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
        assert (env._hd1910_last_push>=0).all()
    finally:
        env.close()
