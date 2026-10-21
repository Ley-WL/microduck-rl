"""Official velocity recipe with observer mechanical bounds and HD1910 physics."""
from .getup_mechanical_contract import LIMITS, edit_mechanical_spec
from mjlab_microduck.robot.microduck_constants import get_walk_spec


def get_observer_walk_spec():
    """Change joint stops only; preserve the walk geometry and contact selection."""
    return edit_mechanical_spec(get_walk_spec())


def apply_observer_contract(cfg):
    """Apply observer's target saturation without changing PPO, rewards or curricula."""
    cfg.scene.entities["robot"].spec_fn = get_observer_walk_spec
    # BaseAction clips after HOME + scale; raw history remains unmodified.
    cfg.actions["joint_pos"].clip = dict(LIMITS)
    return cfg
