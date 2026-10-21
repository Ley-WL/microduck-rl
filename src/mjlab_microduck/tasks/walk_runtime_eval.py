"""Runtime-only HD1910 simulation used by the accepted walking policy evaluation."""
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import numpy as np
import mujoco
import torch
import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.tasks.registry import load_env_cfg
from .getup_mechanical_contract import NAMES, LIMITS, REFERENCES, DIRECTIONS, ENCODER_STEP, edit_mechanical_spec
from . import mdp
from .backlash import make_backlash_variant
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_BACKLASH_ROBOT_CFG
from mjlab_microduck.robot.ground_contact import add_body_ground_contacts
from mjlab_microduck.actuator.feetech_bam import FeetechBamActuator, FeetechBamActuatorCfg

INPUTS=Path(os.environ.get('MICRODUCK_WALK_SURFACE_INPUTS',Path(__file__).resolve().parents[3]/'inputs'))
ROBOT=Path(__file__).resolve().parents[1]/'robot'

def metadata():return json.loads((INPUTS/'walk.metadata.json').read_text())

def robot_spec():
    spec=mujoco.MjSpec.from_file(str(ROBOT/'replica/robot_walk_backlash.xml'))
    add_body_ground_contacts(spec);edit_mechanical_spec(spec)
    # Mass, inertia and assembly transforms come exclusively from the local MJCF.
    return spec

class OutputEncoderActuator(FeetechBamActuator):
    def initialize(self, model, gpu_model, data, device):
        super().initialize(model,gpu_model,data,device)
        names={n:i for i,n in enumerate(self.entity.joint_names)}
        self._passive_ids=torch.tensor([names['passive_'+n+'_backlash'] for n in self._target_names],device=device)
    def get_command(self,data):
        cmd=super().get_command(data)
        return replace(cmd,pos=cmd.pos+data.joint_pos[:,self._passive_ids])

@dataclass(kw_only=True)
class OutputEncoderActuatorCfg(FeetechBamActuatorCfg):
    def build(self,entity,target_ids,target_names):return OutputEncoderActuator(self,entity,target_ids,target_names)

@dataclass(kw_only=True)
class RuntimeWalkActionCfg(JointPositionActionCfg):
    def build(self,env):return RuntimeWalkAction(self,env)

class RuntimeWalkAction(JointPositionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env);assert tuple(self._target_names)==NAMES
        self._home=torch.tensor(np.asarray(metadata()['homeRadians'],dtype=np.float32),device=self.device,dtype=torch.float64)
        self._filtered=torch.zeros_like(self._raw_actions)
        self._initialized=torch.zeros((env.num_envs,1),device=self.device,dtype=torch.bool)
        self._alpha=torch.tensor([.7]*5+[.5]*4+[.7]*5,device=self.device,dtype=torch.float32)
        self._bounds=torch.tensor(list(LIMITS.values()),device=self.device,dtype=torch.float64)
        self._refs=torch.tensor(REFERENCES,device=self.device,dtype=torch.float64)
        self._directions=torch.tensor(DIRECTIONS,device=self.device,dtype=torch.float64)
    def process_actions(self,actions):
        target=.9*actions
        self._filtered=torch.where(self._initialized,(1-self._alpha)*self._filtered+self._alpha*target,target)
        self._initialized.fill_(True);self._raw_actions.copy_(actions)
        bounded=(self._home+self._filtered.double()).clamp(self._bounds[:,0],self._bounds[:,1])
        encoded=torch.floor(self._refs+bounded/ENCODER_STEP*self._directions+.5)
        self._processed_actions=((encoded-self._refs)*ENCODER_STEP*self._directions).float()
    def reset(self,env_ids=None):
        super().reset(env_ids);ids=slice(None) if env_ids is None else env_ids
        self._filtered[ids]=0;self._initialized[ids]=False
    @property
    def position_target(self):return self._processed_actions

def previous_feedback_velocity(env,**params):
    current=mdp.joint_vel_rel_backlash(env,**params)
    step=env.common_step_counter
    if getattr(env,'_walk_velocity_step',None)==step:return env._walk_velocity_observed
    previous=getattr(env,'_walk_velocity_current',current)
    fresh=(env.episode_length_buf==0).unsqueeze(-1)
    env._walk_velocity_observed=torch.where(fresh,current,previous).clone()
    env._walk_velocity_current=current.clone();env._walk_velocity_step=step
    return env._walk_velocity_observed

def encoder_feedback_position(env,biased=False,asset_cfg=mdp._DEFAULT_ASSET_CFG):
    """Actor encoder observation, rounded without target clipping.

    Calibration bias remains training DR; it is applied before encoding.
    The critic retains its privileged continuous position observation.
    """
    robot=env.scene[asset_cfg.name]
    main_ids,passive_ids,mask=mdp._backlash_encoder_ids(env,robot,asset_cfg)
    q=robot.data.joint_pos_biased if biased else robot.data.joint_pos
    position=(q[:,main_ids]+robot.data.joint_pos[:,passive_ids]*mask).double()
    refs=torch.tensor(REFERENCES,device=env.device,dtype=torch.float64)
    directions=torch.tensor(DIRECTIONS,device=env.device,dtype=torch.float64)
    home=torch.tensor(metadata()['homeRadians'],device=env.device,dtype=torch.float64)
    encoded=torch.floor(refs+position/ENCODER_STEP*directions+.5)
    return ((encoded-refs)*ENCODER_STEP*directions-home).float()

def smoothed_twist(env,command_name='twist'):
    current=env.command_manager.get_command(command_name)
    step=env.common_step_counter
    if getattr(env,'_walk_twist_step',None)==step:return env._walk_smoothed_twist
    old=getattr(env,'_walk_smoothed_twist',torch.zeros_like(current))
    old=torch.where((env.episode_length_buf==0).unsqueeze(-1),torch.zeros_like(old),old)
    env._walk_smoothed_twist=old+.2*(current-old);env._walk_twist_step=step
    return env._walk_smoothed_twist

def make_cfg(surface=False,play=False):
    cfg=make_backlash_variant(deepcopy(load_env_cfg('Mjlab-Velocity-Flat-MicroDuck',play=play)),deepcopy(MICRODUCK_WALK_BACKLASH_ROBOT_CFG))
    robot=cfg.scene.entities['robot'];robot.spec_fn=robot_spec
    robot.init_state.joint_pos={**dict(zip(NAMES,metadata()['homeRadians'])),r'^passive_.*':0.}
    robot.collisions=(CollisionCfg(geom_names_expr=(r'^(left|right)_foot_collision$',r'^body_ground_\d+$'),
        contype=1,conaffinity=0,condim=3,priority={r'^(left|right)_foot_collision$':1},friction=(1.,.005,.0001),solref=(.02,1.)),)
    robot.articulation.actuators=(OutputEncoderActuatorCfg(json_path=str(ROBOT/'hd1910/1910_m6.json'),
        target_names_expr=(r'^(?!passive_).*',),kp_fw=5.,vin_range=(7.4,7.4),vin_drop_gain_range=(0.,0.),vin_min=4.,
        delay_min_lag=7,delay_max_lag=7,stiff_frictionloss=True),)
    cfg.actions['joint_pos']=RuntimeWalkActionCfg(**vars(cfg.actions['joint_pos']))
    cfg.actions['joint_pos'].clip=None
    cfg.sim.mujoco.timestep=.002;cfg.decimation=10
    cfg.sim.mujoco.disableflags=('eulerdamp',);cfg.sim.nconmax=200;cfg.sim.contact_sensor_maxmatch=256
    cfg.curriculum.clear()
    # Evaluation context; this does not change the official training recipe.
    for name in ('head_pose','body_pose'):
        cfg.commands[name].ranges=tuple((0.,0.) for _ in cfg.commands[name].ranges)
    for group in ('actor','critic'):
        # Base task already adds one observation-manager delay to velocity.
        # Remove inherited delays before installing the explicit native cache.
        for term in cfg.observations[group].terms.values():
            term.delay_min_lag=0;term.delay_max_lag=0
        cfg.observations[group].terms['joint_vel'].func=previous_feedback_velocity
        cfg.observations[group].terms['command'].func=smoothed_twist
    cfg.observations['actor'].terms['joint_pos'].func=encoder_feedback_position
    cfg.events.pop('push_robot',None)
    cfg.events['foot_friction'].params['ranges']=(1.,1.)
    if surface:
        cfg.scene.terrain.terrain_type='generator'
        cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(size=(4.,4.),border_width=2.,num_rows=2,num_cols=2,
            curriculum=False,sub_terrains={'wave':terrain_gen.HfWaveTerrainCfg(amplitude_range=(.0015,.003),
                num_waves=40,horizontal_scale=.04,vertical_scale=.0005)},add_lights=False)
    return cfg
