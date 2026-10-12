"""Bounded forward-roll transfer; no modifications to deployed policies.

Two roots differ only in a completion-time cost. Both share current mechanical
stops, no EMA, encoder rounding, measured BAM and conservative voltage DR.
"""
from copy import deepcopy
from pathlib import Path
import json
import math
import os
import numpy as np
import torch
from mjlab.managers import RewardTermCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from . import mdp
from .getup_mechanical_contract import LIMITS, NAMES, edit_mechanical_spec
from .microduck_hd1910_getup_mechanical_env_cfg import MechanicalGetupActionCfg
from .microduck_hd1910_skills_env_cfg import make_hd1910_getup_env_cfg, get_skills_spec
from .microduck_roulade_env_cfg import make_microduck_roulade_env_cfg, MicroduckRouladeRlCfg

CASE = Path(__file__).resolve().parents[3]
INPUTS = Path(os.environ.get('MICRODUCK_ROULADE_INPUTS', CASE / 'artifacts/roulade-transfer-20261012/inputs'))

def get_roll_spec():
    spec = edit_mechanical_spec(get_skills_spec())
    contract = json.loads((INPUTS / 'contract.json').read_text())
    bodies = {b.name: b for b in spec.bodies}
    for row in contract['model']['bodies']:
        if row['name'] == 'world':
            continue
        body = bodies[row['name']]
        body.mass = row['mass_kg']
        body.fullinertia = np.asarray([np.nan, 0., 0., 0., 0., 0.])
        body.inertia = np.asarray(row['inertia'])
        body.ipos = np.asarray(row['ipos'])
        body.iquat = np.asarray(row['iquat'])
        body.pos = np.asarray(row['pos'])
        body.quat = np.asarray(row['quat'])
        body.explicitinertial = True
    return spec

def completion_time_cost(accum, head_latch, gravity_z, height):
    # Never tax uprightness during the flip. Finish requires a genuine roll,
    # head pivot and an upright landing; ordinary recovery cannot earn it.
    finished = ((accum >= math.radians(330)) & head_latch &
                (-gravity_z > math.cos(math.radians(20))) & (height > .095))
    return (~finished).float()

def roll_unfinished(env):
    robot = env.scene['robot']
    mdp._update_roulade_accum(env, robot)
    return completion_time_cost(env._roulade_accum, env._roulade_head_latch,
                               robot.data.projected_gravity_b[:, 2],
                               robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2])

def reset_roll_transfer(env, env_ids, real_probability=.20, **params):
    mdp.reset_roulade_state(env, env_ids, **params)
    # Measured start states are coverage, never successful imitation labels.
    if env_ids is None or len(env_ids) == 0:
        return
    if not hasattr(env, '_roll_real_starts'):
        data = np.load(INPUTS / 'real-starts.npz')
        env._roll_real_starts = {k: torch.as_tensor(data[k], device=env.device) for k in data.files}
    ids = env_ids[torch.rand(len(env_ids), device=env.device) < real_probability]
    if not len(ids):
        return
    pool = env._roll_real_starts
    pick = torch.randint(len(pool['qpos']), (len(ids),), device=env.device)
    robot = env.scene['robot']
    servo_ids = mdp._servo_joint_ids(env, robot)
    cols = torch.tensor([7 + j for j in servo_ids], device=env.device)
    env.sim.data.qpos[ids, 2] = .12 + env.scene.env_origins[ids, 2]
    env.sim.data.qpos[ids, 3:7] = pool['quat'][pick].float()
    env.sim.data.qpos[ids[:, None], cols[None]] = pool['qpos'][pick].float()
    env.sim.data.qvel[ids, :3] = 0
    env.sim.data.qvel[ids, 3:6] = pool['gyro'][pick].float()
    vcols = torch.tensor([6 + j for j in servo_ids], device=env.device)
    env.sim.data.qvel[ids[:, None], vcols[None]] = pool['qvel'][pick].float()
    # action_manager.action is the preceding RAW actor output at handoff.
    env.action_manager.action[ids] = pool['previous'][pick].float()
    env._roulade_accum[ids] = 0
    env._roulade_max[ids] = 0
    env._roulade_paid[ids] = 0
    env._roulade_head_latch[ids] = False

def make_roll_transfer_cfg(play=False, repair=True):
    cfg = make_microduck_roulade_env_cfg(play=play)
    cfg.scene.entities['robot'] = deepcopy(make_hd1910_getup_env_cfg(play=play).scene.entities['robot'])
    cfg.scene.entities['robot'].spec_fn = get_roll_spec
    action = MechanicalGetupActionCfg(**vars(cfg.actions['joint_pos']), alpha=0.)
    action.clip = dict(LIMITS)
    cfg.actions['joint_pos'] = action
    actuator = cfg.scene.entities['robot'].articulation.actuators[0]
    actuator.vin_range = (6.8, 8.) if not play else (7.4, 7.4)
    actuator.vin_drop_gain_range = (0., .4) if not play else (0., 0.)
    actuator.vin_min = 5.7
    # This is a scalar supply approximation, not a fitted battery circuit.
    for term in ('base_ang_vel', 'projected_gravity'):
        cfg.observations['actor'].terms[term].delay_min_lag = 0
        cfg.observations['actor'].terms[term].delay_max_lag = 0
    cfg.observations['actor'].terms['base_ang_vel'].func = velocity_mdp.base_ang_vel
    cfg.observations['actor'].terms['base_ang_vel'].params = {}
    cfg.observations['actor'].terms['projected_gravity'].func = velocity_mdp.projected_gravity
    cfg.observations['actor'].terms['projected_gravity'].params = {}
    for axis in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z'):
        setattr(cfg.commands['twist'].ranges, axis, (0., 0.))
    # Fine-tune a learned skill with its final regularizers, not a fresh easy
    # curriculum. Contact/impact/smoothness and speed protections remain.
    for curriculum in cfg.curriculum.values():
        p = curriculum.params
        if 'reward_name' in p and 'weight_stages' in p:
            cfg.rewards[p['reward_name']].weight = p['weight_stages'][-1]['weight']
        if 'event_name' in p and 'param_stages' in p:
            cfg.events[p['event_name']].params.update(p['param_stages'][-1]['params'])
        if 'event_name' in p and 'range_stages' in p:
            end = p['range_stages'][-1]['range']
            cfg.events[p['event_name']].params['ranges'] = (-end, end)
    cfg.curriculum = {}
    cfg.events['set_roulade_state'].func = reset_roll_transfer
    cfg.events['set_roulade_state'].params.update(standing_prob=.8, midroll_prob=.2)
    cfg.rewards['completion_time'] = RewardTermCfg(func=roll_unfinished, weight=-.5 if repair else 0.)
    cfg.sim.nconmax = 256
    cfg.sim.contact_sensor_maxmatch = 256
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    if play:
        cfg.observations['actor'].enable_corruption = False
    return cfg

RollTransferRlCfg = deepcopy(MicroduckRouladeRlCfg)
RollTransferRlCfg.experiment_name = 'roulade_mechanical_transfer'
RollTransferRlCfg.algorithm.learning_rate = 1e-5
RollTransferRlCfg.algorithm.schedule = 'fixed'
RollTransferRlCfg.actor.distribution_cfg['init_std'] = .08
RollTransferRlCfg.max_iterations = 20
RollTransferRlCfg.save_interval = 5
