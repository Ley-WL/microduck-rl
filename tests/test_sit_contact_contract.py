import json
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from mjlab.tasks.registry import load_env_cfg
import mjlab_microduck.tasks
from mjlab_microduck.robot.sit_contact_contract import TARGET_LIMITS
from mjlab_microduck.tasks import mdp


def test_candidate_enables_self_contact_and_target_bounds_after_assembly():
    cfg = load_env_cfg('Mjlab-SitStandContact-Flat-MicroDuck-HD1910')
    robot = cfg.scene.entities['robot']
    spec = robot.spec_fn()
    for collision in robot.collisions:
        collision.edit_spec(spec)
    model = spec.compile()
    geoms = [i for i in range(model.ngeom) if model.geom(i).name.startswith('body_ground_')]
    assert len(geoms) == 68
    assert all((model.geom_contype[g],model.geom_conaffinity[g]) == (3,2) for g in geoms)
    assert cfg.actions['joint_pos'].clip == TARGET_LIMITS
    assert cfg.actions['joint_pos'].alpha == .45
    for name,bounds in TARGET_LIMITS.items():
        np.testing.assert_allclose(model.jnt_range[model.joint(name).id], bounds)
    for name in ('target_overdrive','blocked_target_error'):
        assert cfg.rewards[name].weight < 0
    assert model.nexclude == 1
    assert cfg.events['set_ground_state'].params['sit_overrides'][3] == 1.25
    assert cfg.commands['twist'].sit_z > .065
    for term in cfg.rewards.values():
        if 'sit_overrides' in term.params:
            assert term.params['sit_overrides'][12] == -1.25


def test_blocked_legal_target_costs_more_than_seated_rest():
    term = SimpleNamespace(target_ids=[0,1], position_target=torch.tensor([[1.,1.],[1.,1.]]))
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda _:term),
        scene={'robot': SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor([[1.,1.],[0.,0.]])))})
    cost = mdp.hd1910_sit_target_error_cost(env)
    assert cost[0] == 0 and cost[1] == 16


def test_old_task_contract_is_preserved():
    cfg = load_env_cfg('Mjlab-SitStand-Flat-MicroDuck-HD1910')
    assert cfg.actions['joint_pos'].clip is None
    assert 'blocked_target_error' not in cfg.rewards


def test_objective_repair_keeps_pressure_cost_and_defers_motion_tax():
    cfg = load_env_cfg('Mjlab-SitStandContact-Flat-MicroDuck-HD1910')
    assert cfg.rewards['posture_pose_legs'].func is mdp.posture_pose_group_match
    assert cfg.rewards['posture_height_l1'].params['error_scale'] > .04
    assert cfg.rewards['posture_height_l1'].weight > 0
    assert 'upright_while_tall' not in cfg.rewards
    assert cfg.rewards['upright_linear'].weight > 0
    assert cfg.rewards['blocked_target_error'].weight == -.5
    stages = cfg.curriculum['action_rate_weight'].params['weight_stages']
    assert stages[0] == {'step': 0, 'weight': -.05}
    assert stages[1]['step'] >= 1500 * 24


def test_endpoint_audit_improves_both_command_directions():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('audit', Path(__file__).parents[1] / 'scripts/server/audit_sit_contact_objective.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.audit()['comparison']
    for name, correct, wrong in [('sit_command',1,0), ('stand_command',0,1)]:
        old = result[name]['old']['changed_terms_subtotal']
        new = result[name]['new']['changed_terms_subtotal']
        assert new[correct] - new[wrong] > old[correct] - old[wrong]
        assert new[wrong] < 0 < new[correct]


def test_training_covers_repeated_bidirectional_transitions_without_changing_play():
    cfg = load_env_cfg('Mjlab-SitStandContact-Flat-MicroDuck-HD1910')
    assert cfg.commands['twist'].alternate is True
    assert cfg.commands['twist'].resampling_time_range == (4.,5.)
    assert cfg.episode_length_s >= 24.
    play = load_env_cfg('Mjlab-SitStandContact-Flat-MicroDuck-HD1910',play=True)
    assert play.commands['twist'].alternate is False
    old = load_env_cfg('Mjlab-SitStand-Flat-MicroDuck-HD1910')
    assert old.commands['twist'].alternate is False


def test_alternating_commands_support_partial_resamples_and_random_first_command():
    command=SimpleNamespace(_sit_prob=1.,_alternate=True,device='cpu',
        command_counter=torch.tensor([0,1,2,3]),
        vel_command_b=torch.tensor([[0.,0.,0.],[0.,0.,0.],[1.,0.,0.],[1.,0.,0.]]))
    mdp.SitStandCommand._resample_command(command,torch.tensor([0,1,2]))
    torch.testing.assert_close(command.vel_command_b[:,0],torch.tensor([1.,1.,0.,1.]))
    mdp.SitStandCommand._resample_command(command,torch.tensor([],dtype=torch.long))
    command.command_counter[0]=1
    mdp.SitStandCommand._resample_command(command,torch.tensor([0]))
    assert command.vel_command_b[0,0]==0
