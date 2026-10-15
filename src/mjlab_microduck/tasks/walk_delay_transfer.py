"""Single-factor fixed 16ms control against the fixed 14ms encoder transfer.

This is a simulation delay comparison, not measured UART or firmware latency.
Reward, observation, actor/critic, and actuator physics are inherited unchanged.
"""
from mjlab_microduck.tasks.walk_surface_transfer import make_cfg as encoder_cfg, rl_cfg


def make_cfg(surface=False, play=False):
    cfg = encoder_cfg(surface=surface, play=play)
    for actuator in cfg.scene.entities['robot'].articulation.actuators:
        actuator.delay_min_lag = 8
        actuator.delay_max_lag = 8
    return cfg
