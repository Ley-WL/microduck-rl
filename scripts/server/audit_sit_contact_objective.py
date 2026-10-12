"""Counterfactual endpoint reward audit; no dynamics or hardware access.

Only the three changed terms are summed: this is not total episode reward.
"""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import mjlab_microduck.tasks  # Register tasks before importing robot constants.
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks.microduck_hd1910_sit_contact_env_cfg import (
    get_sit_contact_spec, SIT_OVERRIDES, SIT_Z, STAND_Z,
)
from mjlab_microduck.tasks import mdp


def audit():
    model = get_sit_contact_spec().compile()
    home = torch.tensor([next(v for k, v in HOME_FRAME.joint_pos.items()
        if re.fullmatch(k, model.joint(j).name)) for j in range(1, model.njnt)])
    seated = home.clone()
    for j, value in SIT_OVERRIDES.items():
        seated[j] = value
    q = torch.stack([home, seated])
    asset = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.tensor([[0., 0., STAND_Z], [0., 0., SIT_Z]]),
        root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)))
    command = SimpleNamespace(alpha=torch.ones(2))
    env = SimpleNamespace(scene={'robot': asset}, command_manager=SimpleNamespace(get_term=lambda _: command))
    class Scene(dict):
        terrain = SimpleNamespace(env_origins=torch.zeros(2, 3))
    env.scene = Scene(env.scene)
    params = dict(command_name='twist', sit_overrides=SIT_OVERRIDES,
        joint_indices=[0,1,2,3,4,9,10,11,12,13], std=.5)
    result = {}
    with patch.object(mdp, '_servo_joint_pos', return_value=q), patch.object(
        mdp, '_servo_default_joint_pos', return_value=home.repeat(2, 1)):
        for flag, name in [(1., 'sit_command'), (0., 'stand_command')]:
            command.alpha.fill_(flag)
            height = dict(command_name='twist', sit_z=SIT_Z, stand_z=STAND_Z)
            old = {
                'leg_pose': 4 * mdp.posture_pose_match(env, **params),
                'height_l1': 6 * mdp.posture_height_l1(env, **height),
                'tall_bonus': 1.5 * mdp.upright_while_tall(env, .075, .10),
            }
            new = {
                'leg_pose': 4 * mdp.posture_pose_group_match(env, **params),
                'height_l1': 3 * mdp.posture_height_l1(env, **height, error_scale=STAND_Z-SIT_Z),
                'tall_bonus': torch.zeros(2),
            }
            result[name] = {version: {**{k:v.tolist() for k,v in terms.items()},
                'changed_terms_subtotal': sum(terms.values()).tolist()}
                for version, terms in [('old', old), ('new', new)]}
    return {'state_order':['standing','seated'], 'scope':'three changed reward terms only',
        'physics_revision':'sit-self-contact-bounded-target-v3', 'comparison':result}


if __name__ == '__main__':
    payload = json.dumps(audit(), indent=2)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(payload, encoding='utf-8')
    else:
        print(payload)
