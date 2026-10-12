"""Independent candidate after the 2026-10-10 real sit blocked-load failure.

Do not relabel the old checkpoint as trained with this physics contract.
"""
from copy import deepcopy
import mujoco
from mjlab.managers import RewardTermCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab_microduck.robot.sit_contact_contract import TARGET_LIMITS, enable_sit_self_contacts
from .microduck_hd1910_skills_env_cfg import get_skills_spec, make_hd1910_sitstand_env_cfg, HD1910SitStandRlCfg, SITTING_TARGET_OVERRIDES
from . import mdp

# GPU BAM 64 noisy holds, 5 seconds, last second measured; revised hip support.
# Evidence: artifacts/sit-contact-repair-20261010/gpu-pose-sweep.json.
SIT_Z = .06644204258918762
STAND_Z = .11556478962320947
SIT_OVERRIDES = SITTING_TARGET_OVERRIDES.copy()
SIT_OVERRIDES.update({2:-.4579, 11:.4579, 3:1.25, 12:-1.25})


def get_sit_contact_spec():
    return enable_sit_self_contacts(get_skills_spec())


def make_hd1910_sit_contact_env_cfg(play=False):
    cfg = make_hd1910_sitstand_env_cfg(play=play)
    robot = cfg.scene.entities['robot']
    robot.spec_fn = get_sit_contact_spec
    robot.collisions = (CollisionCfg(
        geom_names_expr=(r'^(left|right)_foot_collision$', r'^body_ground_\d+$'),
        contype=3, conaffinity=2, condim=3,
        priority={r'^(left|right)_foot_collision$': 1},
        friction=(1., .005, .0001), solref={r'^body_ground_\d+$': (.004, 1.)},
    ),)
    # BaseAction clips absolute position AFTER EMA + HOME; raw prev_action stays raw.
    cfg.actions['joint_pos'].clip = TARGET_LIMITS
    cfg.commands['twist'].sit_z, cfg.commands['twist'].stand_z = SIT_Z, STAND_Z
    for term in cfg.rewards.values():
        if 'sit_overrides' in term.params:
            term.params['sit_overrides'] = SIT_OVERRIDES.copy()
        if 'sit_z' in term.params:
            term.params['sit_z'] = SIT_Z
        for name in ('stand_z','target_height','nominal_height'):
            if name in term.params:
                term.params[name] = STAND_Z
    cfg.rewards['rise_bootstrap'].params['max_height'] = STAND_Z + .01
    reset = cfg.events['set_ground_state'].params
    reset['sit_overrides'] = SIT_OVERRIDES.copy()
    # Recompute seated spawn clearance from the revised pose, not old v2.
    model = get_sit_contact_spec().compile()
    data = mujoco.MjData(model)
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME
    from mjlab_microduck.robot.ground_contact import lowest_mesh_z
    import re
    data.qpos[7:] = [next(v for k,v in HOME_FRAME.joint_pos.items() if re.fullmatch(k,model.joint(j).name)) for j in range(1,model.njnt)]
    for index,value in SIT_OVERRIDES.items():
        data.qpos[7+index] = value
    mujoco.mj_forward(model,data)
    heights = list(reset['spawn_heights'])
    heights[1] = float(data.qpos[2]-lowest_mesh_z(model,data))
    reset['spawn_heights'] = tuple(heights)
    cfg.rewards['target_overdrive'] = RewardTermCfg(
        func=mdp.action_over_limit_penalty, weight=-2., params={'overshoot': 0.})
    cfg.rewards['blocked_target_error'] = RewardTermCfg(
        func=mdp.hd1910_sit_target_error_cost, weight=-.5)
    # Objective repair v2: the 1000-iteration actor holds either endpoint but
    # ignores SIT after standing. Preserve physics / deployment contract v3.
    cfg.rewards['posture_pose_legs'].func = mdp.posture_pose_group_match
    cfg.rewards['posture_height_l1'].params['error_scale'] = STAND_Z - SIT_Z
    cfg.rewards['posture_height_l1'].weight = 3.
    # This term rewards being tall even when the command requests sitting.
    # Keep unconditional upright_linear and command-conditioned composite.
    cfg.rewards.pop('upright_while_tall')
    cfg.rewards['action_rate_l2'].weight = -.05
    cfg.curriculum['action_rate_weight'].params['weight_stages'] = [
        {'step': 0, 'weight': -.05},
        {'step': 1500 * 24, 'weight': -.1},
        {'step': 2500 * 24, 'weight': -.2},
    ]
    if not play:
        cfg.commands['twist'].alternate = True
        cfg.commands['twist'].resampling_time_range = (4., 5.)
        cfg.episode_length_s = 24.
    return cfg


HD1910SitContactRlCfg = deepcopy(HD1910SitStandRlCfg)
HD1910SitContactRlCfg.experiment_name = 'hd1910_sitstand_contact_v3'
HD1910SitContactRlCfg.run_name = 'bidirectional_expert_seed'
HD1910SitContactRlCfg.max_iterations = 1000
