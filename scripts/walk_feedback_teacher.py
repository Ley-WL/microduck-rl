"""Privileged simulation-only feedback queries for demonstration generation.

Never install this controller in the robot runtime. The student must learn from
the original command and ordinary 61D observation, not these privileged inputs.
"""
import numpy as np


class FeedbackTeacher:
    def __init__(self, vx_limit=.12):
        if not 0 < vx_limit <= .3:raise ValueError("Invalid simulation vx correction bound")
        self.vx_limit=vx_limit
        self.filtered=np.zeros(2)
        self.integral=np.zeros(2)
        self.correction=np.zeros(2)
        self.previous=np.zeros(2)

    def query(self, command, actual, dt=.02, enabled=True):
        desired=np.asarray(command,dtype=float)[[0,2]]
        actual=np.asarray(actual,dtype=float)
        if not np.isfinite(actual).all() or not np.isfinite(desired).all():
            raise ValueError('Non-finite feedback teacher input')
        if not 0 < dt <= .1:raise ValueError('Invalid feedback period')
        if not enabled or np.linalg.norm(desired)<.001:
            self.filtered[:]=0;self.integral[:]=0;self.correction[:]=0
            self.previous[:]=desired
            return np.asarray(command,dtype=np.float32).copy()
        # Reset integral when either commanded direction reverses.
        reversal=(desired*self.previous<0)&(np.abs(desired)>.02)
        self.integral[reversal]=0;self.correction[reversal]=0
        self.filtered += dt/(.35+dt)*(actual-self.filtered)
        error=desired-self.filtered
        bounds=np.array([self.vx_limit,.6])
        self.integral=np.clip(self.integral+dt*np.array([.8,.8])*error,-bounds,bounds)
        target=np.clip(np.array([.3,.2])*error+self.integral,-bounds,bounds)
        self.correction += np.clip(target-self.correction,-dt*np.array([.3,1.2]),dt*np.array([.3,1.2]))
        result=np.asarray(command,dtype=np.float32).copy()
        result[[0,2]]=np.clip(desired+self.correction,[-.3,-1.2],[.3,1.2])
        self.previous[:]=desired
        return result


class TensorFeedbackTeacher:
    """Batched equivalent for GPU demonstration generation; no runtime use."""
    def __init__(self):
        self.filtered=None

    def query(self, command, actual, dt=.02, enabled=None):
        import torch
        desired=command[:,[0,2]]
        if self.filtered is None:
            self.filtered=torch.zeros_like(desired)
            self.integral=torch.zeros_like(desired)
            self.correction=torch.zeros_like(desired)
            self.previous=torch.zeros_like(desired)
        if not 0 < dt <= .1:raise ValueError('Invalid feedback period')
        active=desired.norm(dim=-1)>=.001
        if enabled is not None:active=active & enabled
        reversal=(desired*self.previous<0)&(desired.abs()>.02)
        self.integral=torch.where(reversal,0.,self.integral)
        self.correction=torch.where(reversal,0.,self.correction)
        self.filtered=self.filtered+dt/(.35+dt)*(actual-self.filtered)
        error=desired-self.filtered
        bounds=desired.new_tensor([.12,.6])
        self.integral=(self.integral+dt*.8*error).clamp(-bounds,bounds)
        target=(desired.new_tensor([.3,.2])*error+self.integral).clamp(-bounds,bounds)
        slew=dt*desired.new_tensor([.3,1.2])
        self.correction=self.correction+(target-self.correction).clamp(-slew,slew)
        result=command.clone()
        corrected=(desired+self.correction).clamp(desired.new_tensor([-.3,-1.2]),desired.new_tensor([.3,1.2]))
        result[:,[0,2]]=torch.where(active[:,None],corrected,desired)
        for name in ('filtered','integral','correction'):
            setattr(self,name,torch.where(active[:,None],getattr(self,name),0.))
        self.previous=desired.clone()
        return result
