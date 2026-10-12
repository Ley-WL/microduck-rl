"""Target the measured low-gain sensitivity frontier without changing motors.

Five equal buckets: minimum, near-minimum, nominal, maximum, and continuous.
The lower mean is intentional; this is a distribution intervention, not an
identified physical P/D setting or proof that rarity alone caused failure.
"""
import torch


def sample_boundary_kp(worlds,device,kp_range=(.8,1.2)):
    low,high=kp_range
    assert 0<low<high
    bucket=torch.randint(5,(worlds,1),device=device)
    values=torch.empty((worlds,1),device=device).uniform_(low,high)
    for index,value in enumerate((low,low+.05*(high-low),(low+high)/2,high)):
        values=torch.where(bucket==index,value,values)
    return values,bucket


def randomize_boundary_gains(env,env_ids,kp_range=(.8,1.2),kd_range=(.9,1.1)):
    from bam.mjlab import BamActuator
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    for actuator in env.scene['robot'].actuators:
        if not isinstance(actuator,BamActuator):continue
        kp,_=sample_boundary_kp(len(env_ids),env.device,kp_range)
        kd=torch.empty((len(env_ids),1),device=env.device).uniform_(*kd_range)
        actuator.reset_gains(env_ids)
        actuator.set_gains(env_ids,kp_scale=kp,kd_scale=kd)
