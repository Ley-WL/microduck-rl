"""Matched short controls; vary one objective/observation factor at a time."""
from copy import deepcopy
from mjlab.managers import RewardTermCfg
from .microduck_hd1910_sit_process_env_cfg import make_hd1910_sit_process_env_cfg,HD1910SitProcessRlCfg
from . import mdp,sit_process_cost

CASES=('Coverage','WeakProcess','Temporal','Sensor','StateReference','Upright','UprightCapped','LowExplore','CriticWarm','ArrivalGate','BoundaryGain','FullBatch')


def make_sit_ablation_env_cfg(case,play=False):
    assert case in CASES
    cfg=make_hd1910_sit_process_env_cfg(play)
    cfg.rewards.pop('knee_process')
    cfg.rewards['target_acceleration']=RewardTermCfg(func=mdp.hd1910_target_acceleration_cost,weight=-.05)
    if case in ('WeakProcess','Sensor','StateReference','Upright','UprightCapped','LowExplore','CriticWarm','ArrivalGate','BoundaryGain','FullBatch'):
        cfg.rewards['knee_process']=RewardTermCfg(func=sit_process_cost.knee_process_cost,weight=-.015,params={'scale_rad':.05})
    if case in ('Upright','UprightCapped'):
        from .sit_settled_tilt import settled_tilt_cost
        cfg.rewards['settled_tilt']=RewardTermCfg(func=settled_tilt_cost,weight=-2.)
        if case=='UprightCapped':
            cfg.rewards['settled_tilt'].params['max_cost']=4.
    if case=='ArrivalGate':
        from . import sit_arrival_gate as gate
        cfg.rewards['posture_pose_legs'].func=gate.arrival_pose_match
        for name in ('posture_height','posture_height_sharp'):
            cfg.rewards[name].func=gate.arrival_height_match
        cfg.rewards['posture_stillness'].func=gate.arrival_stillness
    if case=='BoundaryGain' and not play:
        from .sit_boundary_gains import randomize_boundary_gains
        cfg.events['transfer_motor_gains'].func=randomize_boundary_gains
    if case=='StateReference':
        from .sit_state_reference import StateReferenceCommand,StateReferenceCommandCfg
        cfg.commands['twist']=StateReferenceCommandCfg(**vars(cfg.commands['twist']))
        cfg.commands['twist'].class_type=StateReferenceCommand
    if case=='Sensor' and not play:
        # Independent gyro corruption hypothesis, not measured sensor noise.
        noise=cfg.observations['actor'].terms['base_ang_vel'].noise
        noise.n_min,noise.n_max=-.5,.5
    return cfg


def make_sit_ablation_rl_cfg(case):
    cfg=deepcopy(HD1910SitProcessRlCfg)
    cfg.algorithm.temporal_coef=3. if case=='Temporal' else .3
    cfg.algorithm.onpolicy_retention_coef=2.
    if case=='FullBatch':cfg.algorithm.num_mini_batches=1
    if case=='LowExplore':cfg.algorithm.exploration_std_range=(.02,.03)
    cfg.experiment_name='hd1910_sit_process_ablation'
    cfg.run_name=case.lower()+'100'
    cfg.max_iterations=100
    cfg.save_interval=100
    if case=='CriticWarm':
        from .sit_critic_warm import SitCriticWarmCfg
        cfg.algorithm=SitCriticWarmCfg(**{k:v for k,v in vars(cfg.algorithm).items() if k!='class_name'})
        cfg.max_iterations=120  # 20 critic-only rollouts + matched 100 actor updates.
    return cfg
