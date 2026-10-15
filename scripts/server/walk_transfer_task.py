"""Explicit task selector shared by prepare/train scripts; old task is default."""
import importlib
import os

TASK = os.environ.get('MICRODUCK_WALK_TRANSFER_TASK', 'walk_stop_transfer')
assert TASK in ('walk_stop_transfer', 'walk_tracking_transfer', 'walk_stride_transfer', 'walk_sustained_transfer'), TASK
module = importlib.import_module('mjlab_microduck.tasks.' + TASK)
make_cfg, rl_cfg = module.make_cfg, module.rl_cfg
