import importlib.util
from pathlib import Path

import numpy as np
import mujoco
import pytest

spec = importlib.util.spec_from_file_location('skill_eval', Path(__file__).parents[1]/'scripts/server/evaluate_skills.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_gate_diagnostics_do_not_relax_the_combined_success_criterion():
    # Every defective row violates one different requirement.
    tilt = np.array([0., 30., 0., 0., 0., 0.])
    height = np.array([evaluation.STAND_Z]*6)
    height[2] -= .02
    speed = np.array([0., 0., 0., .3, 0., 0.])
    angular = np.array([0., 0., 0., 0., 3., 0.])
    leg_error = np.array([0., 0., 0., 0., 0., .3])
    gates = evaluation.stability_gates(tilt, height, speed, angular, leg_error, False)
    np.testing.assert_array_equal(evaluation.stable_mask(tilt, height, speed, angular, leg_error, False), [True, False, False, False, False, False])
    assert all((~gate).sum() == 1 for gate in gates.values())


def test_sitting_cannot_pass_using_standing_height():
    values = np.array([evaluation.SIT_Z, evaluation.STAND_Z])
    zeros = np.zeros(2)
    np.testing.assert_array_equal(evaluation.stable_mask(zeros, values, zeros, zeros, zeros, True), [True, False])


def test_nonfinite_observed_state_cannot_pass():
    zeros = np.zeros(2)
    heights = np.array([np.nan, np.inf])
    assert not evaluation.stable_mask(zeros, heights, zeros, zeros, zeros, False).any()


def test_duplicate_successful_scenario_cannot_replace_missing_scenario():
    rows = [{'scenario': 'prone', 'seed': 1009}] * 2
    assert not evaluation.scope_complete(rows, ['prone', 'supine'], [1009])


def test_scope_requires_every_scenario_seed_pair():
    rows = [{'scenario': scenario, 'seed': seed}
            for scenario in ['prone', 'supine'] for seed in [1009, 2027]]
    assert evaluation.scope_complete(rows, ['prone', 'supine'], [1009, 2027])
    assert not evaluation.scope_complete(rows[:-1], ['prone', 'supine'], [1009, 2027])


@pytest.mark.parametrize('contype,conaffinity,valid', [(1, 0, True), (0, 0, False), (1, 1, False)])
def test_acceptance_checks_assembled_contact_masks(contype, conaffinity, valid):
    geoms = ''.join(f'<geom name="robot/body_ground_{i}" size=".01" contype="{contype}" conaffinity="{conaffinity}" condim="3"/>'
                    for i in range(68))
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="terrain" type="plane" size="1 1 .1" contype="1" conaffinity="1"/>'
        + geoms + '</worldbody></mujoco>')
    if valid:
        evaluation.verify_assembled_skill_contacts(model)
    else:
        with pytest.raises(ValueError, match='contact contract'):
            evaluation.verify_assembled_skill_contacts(model)
