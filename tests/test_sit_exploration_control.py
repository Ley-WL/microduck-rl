"""Exploration changes sampling variance, never the deployed actor mean."""
from types import SimpleNamespace
from dataclasses import fields,is_dataclass
import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab_microduck.tasks.sit_transfer_ppo import clamp_exploration_std


def test_loaded_std_is_bounded_before_first_rollout_and_after_updates():
    distribution=SimpleNamespace(std_param=torch.nn.Parameter(torch.tensor([.1,.01,.2])))
    clamp_exploration_std(distribution,(.02,.03))
    torch.testing.assert_close(distribution.std_param,torch.tensor([.03,.02,.03]))


def test_low_exploration_preserves_full_environment_and_other_ppo_settings():
    old='Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910'
    new='Mjlab-SitStandProcessLowExplore-Flat-MicroDuck-HD1910'
    # Each scene constructs a fresh default terrain lambda; compare its
    # qualified name rather than Python object identity.
    def canonical(value):
        if is_dataclass(value):return {f.name:canonical(getattr(value,f.name)) for f in fields(value)}
        if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return tuple(canonical(v) for v in value)
        if callable(value):return (value.__module__,value.__qualname__)
        return value
    assert canonical(load_env_cfg(old))==canonical(load_env_cfg(new))
    a,b=load_rl_cfg(old),load_rl_cfg(new)
    assert b.algorithm.exploration_std_range==(.02,.03)
    a.algorithm.exploration_std_range=b.algorithm.exploration_std_range
    assert a.algorithm==b.algorithm
