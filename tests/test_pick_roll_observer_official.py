"""Lock public rewards, curricula, PPO, symmetry and physical limits."""
import pytest
from dataclasses import asdict
from mjlab_microduck.tasks.pick_roll_observer_official import FACTORIES, RUNNERS, make_cfg, rl_cfg
from mjlab_microduck.tasks.getup_mechanical_contract import LIMITS


@pytest.mark.parametrize('task', ['pick', 'roll'])
def test_task_recipe(task):
    actual, original = make_cfg(task), FACTORIES[task]()
    for key in ('rewards', 'curriculum', 'commands', 'terminations', 'events', 'observations'):
        assert repr(getattr(actual, key)) == repr(getattr(original, key)), key
    assert actual.episode_length_s == original.episode_length_s


@pytest.mark.parametrize('task', ['pick', 'roll'])
def test_ppo(task):
    actual, original = asdict(rl_cfg(task)), asdict(RUNNERS[task])
    for key in ('actor', 'critic', 'algorithm', 'num_steps_per_env', 'max_iterations', 'save_interval'):
        assert actual[key] == original[key], key
    assert actual['max_iterations'] == (20000 if task == 'pick' else 10000)


@pytest.mark.parametrize('task', ['pick', 'roll'])
def test_limits(task):
    cfg = make_cfg(task)
    assert cfg.actions['joint_pos'].clip == LIMITS
    spec = cfg.scene.entities['robot'].spec_fn()
    found = {j.name: tuple(j.range) for j in spec.joints if j.name in LIMITS}
    assert found == LIMITS
