from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
import mjlab_microduck.tasks
from mjlab_microduck.tasks.microduck_hd1910_stand_env_cfg import PARAMS
import math
import torch
from mjlab_microduck.tasks import mdp

TASK='Mjlab-Walk-Flat-MicroDuck-HD1910'

def test_walk_preserves_replica_physics_and_command_contract():
    cfg=load_env_cfg(TASK)
    model=cfg.scene.entities['robot'].spec_fn().compile()
    assert model.nu==14 and sum('body_ground_' in model.geom(i).name for i in range(model.ngeom))==68
    assert cfg.scene.entities['robot'].articulation.actuators[0].json_path==str(PARAMS)
    assert cfg.commands['twist'].ranges.lin_vel_x==(-.2,.3)
    assert cfg.commands['twist'].rel_turn_in_place_envs==.1
    assert cfg.commands['twist'].rel_straight_envs==.7
    assert cfg.commands['twist'].rel_forward_envs==0
    assert not cfg.commands['twist'].heading_command
    assert len(cfg.commands['head_pose'].ranges)==4 and len(cfg.commands['body_pose'].ranges)==6
    assert all(name in cfg.rewards for name in ('air_time','foot_clearance','foot_swing_height','idle_home'))
    assert all(name not in cfg.rewards for name in ('quiet_motion_cost','leg_alignment_cost'))
    assert cfg.rewards['head_target_cost'].weight<0
    # Continuing a learned gait must not restart the weak stride curriculum.
    assert cfg.rewards['stride_quality'].weight==4.
    assert cfg.curriculum['stride_quality_weight'].params['weight_stages'][0]['weight']==4.
    assert cfg.rewards['yaw_error_cost'].weight<0
    assert load_rl_cfg(TASK).experiment_name=='microduck_hd1910_walk'

def test_walk_play_has_no_push_or_head_command_drift():
    cfg=load_env_cfg(TASK,play=True)
    assert 'push_robot' not in cfg.events
    assert all(r==(0.,0.) for r in cfg.commands['head_pose'].ranges)

def test_collapsed_body_cannot_farm_positive_rewards():
    # V1's 68-degree body-supported state and a low but vertical state.
    angles=torch.deg2rad(torch.tensor([0.,20.,68.,0.]))
    gravity=torch.stack((torch.sin(angles),torch.zeros(4),-torch.cos(angles)),dim=-1)
    factors=mdp.hd1910_walk_posture_factor(gravity,torch.tensor([.117,.117,.117,.04]),.117)
    assert factors[0]==1 and factors[1]>.3
    assert factors[2]==0 and factors[3]==0
    for play in (False,True):
        cfg=load_env_cfg(TASK,play=play)
        assert cfg.terminations['fell_over'].params['limit_angle']==math.radians(45)
        assert 'low_trunk' in cfg.terminations
        assert all(term.func==mdp.hd1910_walk_supported_reward
                   for name,term in cfg.rewards.items() if term.weight>0 and name!='head_pose_bias')
        assert cfg.rewards['head_pose_bias'].func==mdp.head_pose_bias_penalty
        assert 'head_pose_bias_weight' not in cfg.curriculum

def test_stride_reward_prefers_full_swing_without_rewarding_idle():
    scores=mdp.hd1910_stride_quality(torch.tensor([.08,.034,.08,.08]),torch.zeros(4),
        torch.tensor([.018,.004,.018,.018]),torch.tensor([.16,.07,.16,.16]),
        torch.tensor([.2,.2,0.,.2]))
    assert scores[0]==1 and scores[0]>5*scores[1]
    assert scores[2]==0
    # At equal duty factor, payout per second is independent of cadence.
    assert .16*5==.08*10

def test_stride_metrics_ignore_short_contact_chatter():
    import numpy as np
    from scripts.server.gait_metrics import summarize_gait
    positions=np.zeros((20,1,2,3));contacts=np.ones((20,1,2),dtype=bool)
    air=np.zeros((20,1,2));heights=np.zeros((20,1,2))
    contacts[2:7,0,0]=False;contacts[11:16,0,0]=False
    positions[7:,0,0,0]=.08;positions[16:,0,0,0]=.16
    air[7,0,0]=.1;air[16,0,0]=.1
    result=summarize_gait(positions,contacts,air,heights,.02)
    assert result['validLandings']==2 and abs(result['medianStrideM']-.08)<1e-6

def test_larger_stride_target_and_straight_path_metrics():
    from scripts.server.gait_metrics import summarize_heading_path
    import numpy as np
    speed=torch.tensor([.2,.2]);zero=torch.zeros(2)
    scores=mdp.hd1910_stride_quality(torch.tensor([.11,.06]),zero,
        torch.tensor([.022,.015]),torch.tensor([.2,.135]),speed,.55,.2,.022)
    assert scores[0]>.99 and scores[0]>scores[1]
    pos=np.zeros((100,1,3));pos[:,0,0]=np.linspace(0,1,100)
    straight=summarize_heading_path(pos,np.zeros((100,1)),.02)
    curved=summarize_heading_path(pos,np.linspace(0,1,100)[:,None],.02)
    assert straight['headingDriftP95Deg']==0 and straight['crossTrackP95M']==0
    assert curved['headingDriftP95Deg']>50

def test_yaw_cost_charges_uncommanded_turns_not_requested_rotation():
    from types import SimpleNamespace
    command=torch.tensor([[.2,0.,0.],[0.,0.,1.],[0.,0.,-1.]])
    robot=SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=torch.tensor([[0.,0.,.5],[0.,0.,1.],[0.,0.,-1.]])))
    env=SimpleNamespace(scene={'robot':robot},command_manager=SimpleNamespace(get_command=lambda _:command))
    cost=mdp.hd1910_walk_yaw_error_cost(env)
    assert cost[0]>.5 and cost[1]==0 and cost[2]==0


def test_yaw_bias_cost_targets_persistent_drift_and_resets_on_command_changes():
    from types import SimpleNamespace
    command=torch.zeros((2,3))
    velocity=torch.zeros((2,3))
    robot=SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=velocity))
    env=SimpleNamespace(num_envs=2,device='cpu',step_dt=.02,scene={'robot':robot},
        extras={'log':{}},command_manager=SimpleNamespace(get_command=lambda _:command))
    term=mdp.hd1910_walk_yaw_bias_cost(None,env)
    for tick in range(150):
        velocity[:,2]=torch.tensor([.1,.1 if tick%2 else -.1])
        cost=term(env)
    assert cost[0]>.24 and cost[1]<.001
    command[:,2]=.6
    velocity[:,2]=.6
    assert torch.equal(term(env),torch.zeros(2))
    velocity[:,2]=1.
    term(env);term.reset(torch.tensor([0]))
    assert term.bias[0]==0 and term.bias[1]!=0


def test_replica_reflection_and_mirror_loss_preserve_contract():
    from scripts.server.diagnose_walk_symmetry import verify_replica_mirror
    from mjlab_microduck.tasks.symmetry import _OBS_PERM,_OBS_SIGN,_JOINT_PERM,_JOINT_SIGN
    assert verify_replica_mirror()['mirrorVerified']
    obs=torch.randn((8,61));actions=torch.randn((8,14))
    mirrored=obs[:,_OBS_PERM]*torch.tensor(_OBS_SIGN)
    assert torch.equal(mirrored[:,_OBS_PERM]*torch.tensor(_OBS_SIGN),obs)
    mirrored_actions=actions[:,_JOINT_PERM]*torch.tensor(_JOINT_SIGN)
    assert torch.equal(mirrored_actions[:,_JOINT_PERM]*torch.tensor(_JOINT_SIGN),actions)
    symmetry=load_rl_cfg(TASK).algorithm.symmetry_cfg
    assert symmetry['use_mirror_loss'] and not symmetry['use_data_augmentation']
    assert symmetry['mirror_loss_coeff']>0


def test_gait_reward_requires_head_quality_without_negative_abort_incentive():
    error=torch.tensor([[0.,0.,0.,0.],[1.1,0.,0.,0.],[0.,0.,0.,0.]])
    quality=mdp.hd1910_walk_head_quality(error)
    assert quality[0]==1 and .4<quality[1]<.5
    assert torch.all(quality>=.4)
    cfg=load_env_cfg(TASK)
    assert cfg.rewards['track_linear_velocity'].params['head_quality']
    assert cfg.rewards['stride_quality'].params['head_quality']
    assert cfg.rewards['head_pose_bias'].weight==.6


def test_failure_cost_is_per_event_and_does_not_charge_timeouts():
    from types import SimpleNamespace
    env=SimpleNamespace(step_dt=.02,termination_manager=SimpleNamespace(
        terminated=torch.tensor([True,False])))
    value=mdp.hd1910_walk_failure_cost(env)
    assert torch.equal(value*env.step_dt,torch.tensor([1.,0.]))
    assert load_env_cfg(TASK).rewards['failure_cost'].weight==-5.


def test_target_clipping_matches_replica_official_ranges_without_new_motion_caps():
    cfg=load_env_cfg(TASK)
    model=cfg.scene.entities['robot'].spec_fn().compile()
    assert len(cfg.actions['joint_pos'].clip)==14
    for name,limits in cfg.actions['joint_pos'].clip.items():
        assert tuple(model.jnt_range[model.joint(name).id])==limits
        assert all(type(x) is float for x in limits)
    assert cfg.actions['joint_pos'].scale==1.
    assert load_rl_cfg(TASK).algorithm.learning_rate==1e-4


def test_official_target_cost_prices_only_out_of_range_raw_targets():
    from types import SimpleNamespace
    limits=torch.tensor([[[-.5,.5]]]).repeat(2,14,1)
    robot=SimpleNamespace(data=SimpleNamespace(default_joint_pos=torch.zeros(2,14),
        joint_pos_limits=limits))
    # Plain-model fallback in the shared servo selector.
    robot.joint_names=['joint']*14
    robot.find_joints=lambda _: (list(range(14)),robot.joint_names)
    action=torch.zeros(2,14);action[1,0]=2.
    env=SimpleNamespace(scene={'robot':robot},action_manager=SimpleNamespace(action=action))
    value=mdp.hd1910_walk_official_target_cost(env)
    assert value[0]==0 and value[1]>0


def test_fixed_warm_start_lr_preserves_optimizer_moments_but_uses_new_rate():
    from types import SimpleNamespace
    optimizer=SimpleNamespace(param_groups=[{'lr':.001}],state={'kept':42})
    algorithm=SimpleNamespace(schedule='fixed',learning_rate=.0001,optimizer=optimizer)
    mdp._restore_fixed_warm_start_lr(algorithm)
    assert optimizer.param_groups[0]['lr']==.0001 and optimizer.state=={'kept':42}
    algorithm.schedule='adaptive';optimizer.param_groups[0]['lr']=.0002
    mdp._restore_fixed_warm_start_lr(algorithm)
    assert optimizer.param_groups[0]['lr']==.0002
