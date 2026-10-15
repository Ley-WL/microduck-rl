"""Explicit task selector shared by prepare/train scripts; old task is default."""
import importlib
import os

TASK = os.environ.get('MICRODUCK_WALK_TRANSFER_TASK', 'walk_stop_transfer')
assert TASK in ('walk_stop_transfer', 'walk_tracking_transfer', 'walk_stride_transfer', 'walk_sustained_transfer', 'walk_low_speed_stride', 'walk_low_speed_control', 'walk_low_speed_focused', 'walk_low_speed_focused_control','walk_low_speed_air_window','walk_low_speed_mean_tracking','walk_mean_motion_tracking','walk_motion_critic_context','walk_continuous_coverage','walk_relative_cycle_context'), TASK
module = importlib.import_module('mjlab_microduck.tasks.' + TASK)
make_cfg, rl_cfg = module.make_cfg, module.rl_cfg
