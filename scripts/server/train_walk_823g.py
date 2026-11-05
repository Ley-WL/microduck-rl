"""Fine-tune the Observer 5000 walking policy in two stages on the 823 g replica."""

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab_microduck.tasks import MicroduckOnPolicyRunner
from mjlab_microduck.tasks.walk_mass_823g import apply_mass_823g
from mjlab_microduck.tasks.walk_observer_contract import apply_observer_contract, get_observer_walk_spec

TASK = "Mjlab-Velocity-Flat-MicroDuck"
BASELINE = Path(__file__).resolve().parents[2] / "models/walk_observer_hd1910_5000/policy.onnx"
BASELINE_SHA256 = "e3b90ac0e5359f91e14306b7e2143e0d5a1f4cd49cb256bca3fd71835250d895"


def warm_start(actor, policy, device):
    tensors = {v.name: onnx.numpy_helper.to_array(v).copy() for v in onnx.load(policy).graph.initializer}
    state = actor.state_dict()
    for name in state.keys() & tensors.keys():
        state[name] = torch.as_tensor(tensors[name], device=device)
    divisor = next(v for name, v in tensors.items() if "Div" in name)
    std = torch.as_tensor(divisor, device=device) - 0.01
    state["obs_normalizer._std"] = std
    state["obs_normalizer._var"] = std.square()
    state["obs_normalizer.count"].fill_(1_000_000)
    state["distribution.std_param"].fill_(0.10)
    actor.load_state_dict(state)


def verify_actor(actor, policy, device):
    ref = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    rows = np.random.default_rng(211).normal(0, 0.35, (128, 61)).astype("float32")
    actor.eval()
    with torch.no_grad():
        actual = actor(TensorDict({"actor": torch.tensor(rows, device=device)}, batch_size=[128]))
    expected = np.concatenate([ref.run(None, {ref.get_inputs()[0].name: row[None]})[0] for row in rows])
    error = np.abs(actual.cpu().numpy() - expected).max()
    assert error < 2e-5, error
    actor.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["warmup", "continue"])
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    warmup = args.stage == "warmup"
    first = args.runs / "walk823-logs"
    output = first if warmup else args.runs / "walk823-plus10000"
    policy = BASELINE if warmup else first / "walk823_final.onnx"
    if warmup:
        assert hashlib.sha256(policy.read_bytes()).hexdigest() == BASELINE_SHA256
    output.mkdir(parents=True, exist_ok=False)
    cfg = apply_observer_contract(load_env_cfg(TASK))
    cfg.scene.entities["robot"].spec_fn = lambda: apply_mass_823g(get_observer_walk_spec())
    cfg.scene.num_envs = 4096
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        rl = asdict(load_rl_cfg(TASK))
        rl.update(logger="tensorboard", upload_model=False)
        runner = MicroduckOnPolicyRunner(RslRlVecEnvWrapper(env), rl, log_dir=str(output), device=env.device)
        if warmup:
            warm_start(runner.alg.actor, policy, env.device)
            env.common_step_counter = 5000 * 24
        else:
            runner.load(str(first / "model_999.pt"))
            assert runner.alg.optimizer.state and runner.current_learning_iteration == 999
            assert env.common_step_counter == 6000 * 24
            runner.current_learning_iteration = 1000
        verify_actor(runner.alg.actor, policy, env.device)
        runner.learn(1000 if warmup else 10000, init_at_random_ep_len=True)
        assert env.common_step_counter == (6000 if warmup else 16000) * 24
        runner.export_policy_to_onnx(str(output), "walk823_final.onnx")
    finally:
        env.close()


if __name__ == "__main__":
    main()
