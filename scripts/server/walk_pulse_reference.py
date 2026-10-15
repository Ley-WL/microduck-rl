"""Offline pulse reference only; original command and actor input contract unchanged."""
import torch

class PulseReference:
    def __init__(self,worlds,device):
        self.query=torch.zeros((worlds,3),device=device)
    def step(self,command,elapsed):
        period=.1/command[:,0].abs().clamp_min(.035)
        active=command[:,0].abs()>.001
        target=torch.zeros_like(command)
        # Quantize the accumulated float clock to policy ticks: a nominal .4s
        # pulse must not gain a 21st frame from float32 boundary roundoff.
        phase_ticks=(elapsed.clamp_min(0)/.02).round().remainder(period/.02)
        target[:,0]=torch.where(active & (phase_ticks<20.-1e-4),command[:,0].sign()*.2,0.)
        self.query.lerp_(target,.2)
        return self.query.clone()
    def reset(self,mask):
        self.query[mask]=0.
