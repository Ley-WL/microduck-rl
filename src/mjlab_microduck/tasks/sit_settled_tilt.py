"""Endpoint tilt cost; leave necessary transitional leaning unpriced here."""
import math
import torch


def settled_tilt_values(gravity, flag, blend, free_degrees=12., scale_degrees=10., max_cost=None):
    # atan2 is well behaved near upright, unlike acos gradients at ±1.
    tilt=torch.atan2(gravity[:,:2].norm(dim=-1),-gravity[:,2])
    excess=(tilt-math.radians(free_degrees)).clamp_min(0.)/math.radians(scale_degrees)
    cost=excess.square()
    if max_cost is not None:
        assert max_cost>0
        cost=cost.clamp_max(max_cost)
    return cost * ((flag-blend).abs()<.02)


def settled_tilt_cost(env,command_name='twist',max_cost=None):
    term=env.command_manager.get_term(command_name)
    return settled_tilt_values(env.scene['robot'].data.projected_gravity_b,
                               term.command[:,0],term.alpha,max_cost=max_cost)
