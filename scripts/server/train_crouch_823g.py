"""Train the foot-supported roller-style crouch and rise on the 823 g replica."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab_microduck.tasks import MicroduckOnPolicyRunner, sitstand_observer_official as base
from mjlab_microduck.tasks.mdp import posture_composite
from mjlab_microduck.tasks.walk_mass_823g import apply_mass_823g

MODEL = Path(__file__).resolve().parents[2] / "models/crouch_hd1910_823g_3000"


def supported_posture(env, **kwargs):
    score = posture_composite(env, **kwargs)
    contacts = env.scene.sensors["feet_ground_contact"].data.found.reshape(env.num_envs, -1)
    return score * (contacts > 0).all(dim=1)


def crouch_spec():
    return apply_mass_823g(base.spec())


def make_cfg():
    target = json.loads((MODEL / "target.json").read_text())
    angles = dict(enumerate(target["q"]))
    height = target["z"]
    cfg = base.make_cfg()
    cfg.scene.entities["robot"].spec_fn = crouch_spec
    cfg.scene.num_envs = 4096
    for term in cfg.rewards.values():
        params = term.params
        if "sit_overrides" in params:
            params["sit_overrides"] = angles
            if "joint_indices" in params:
                params["joint_indices"] = list(range(14))
        if "sit_z" in params:
            params["sit_z"] = height
    del cfg.rewards["head_pose_tracking"]
    head = cfg.commands["head_pose"]
    head.ranges = tuple((0., 0.) for _ in head.ranges)
    cfg.curriculum.pop("head_pose_range", None)
    posture = cfg.rewards["posture_composite"]
    posture.func = supported_posture
    posture.weight = 6.
    posture.params["head_std"] = None
    cfg.rewards["posture_stillness"].weight *= 1.5
    cfg.events["set_ground_state"].params.update(
        sitting_joint_overrides=angles, sitting_z_min=height + .002, sitting_z_max=height + .012
    )
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    env = ManagerBasedRlEnv(make_cfg(), device="cuda:0")
    try:
        runner = MicroduckOnPolicyRunner(
            RslRlVecEnvWrapper(env), asdict(base.rl_cfg()), log_dir=str(out), device=env.device
        )
        runner.learn(3000, init_at_random_ep_len=True)
        runner.export_policy_to_onnx(str(out), "policy.onnx")
    finally:
        env.close()


if __name__ == "__main__":
    main()
