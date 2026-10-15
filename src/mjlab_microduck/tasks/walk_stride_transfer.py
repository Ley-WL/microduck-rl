"""More observable long-step credit, same physics and control contract as v1."""
from . import walk_tracking_transfer as base


def make_cfg(surface=False, play=False):
    cfg = base.make_cfg(surface, play)
    term = cfg.rewards['long_stride']
    term.weight = 6.
    # The triple squared completion product erased useful gradient for existing
    # 3-5 cm steps. Component shaping already increases with distance, clearance
    # and air time. Keep touchdown-only, alternation and air-time weighting.
    term.params['complete_swing'] = False
    return cfg


def rl_cfg():
    cfg = base.rl_cfg()
    cfg.max_iterations = 600
    cfg.experiment_name = 'walk_stride_progressive'
    return cfg
