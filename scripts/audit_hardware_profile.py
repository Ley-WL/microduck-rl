"""Verify local model inputs, optionally resolve every registered training task."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mjlab_microduck.robot.hardware_profile import (
    HARDWARE_PROFILE, HD1910_PARAMS, ROBOT_DIR, ROBOT_ROOT, verify_inputs,
)


def audit_inputs():
    count = verify_inputs()
    params = json.loads(HD1910_PARAMS.read_text(encoding="utf-8"))
    assert params["actuator"] == "sts3215" and params["model"] == "m6"
    return {"profile": HARDWARE_PROFILE, "verifiedFiles": count}


def audit_tasks():
    import mujoco
    import mjlab_microduck.tasks
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
    from mjlab_microduck.actuator.feetech_bam import FeetechBamActuatorCfg

    rows = []
    cache = {}
    for name in list_tasks():
        if "MicroDuck" not in name:
            continue
        agent = load_rl_cfg(name)
        assert getattr(agent.algorithm, "bc_cfg", None) is None, name
        for play in (False, True):
            cfg = load_env_cfg(name, play=play)
            robot = cfg.scene.entities["robot"]
            for actuator in robot.articulation.actuators:
                assert isinstance(actuator, FeetechBamActuatorCfg), name
                assert actuator.motor_name is None, name
                assert Path(actuator.json_path).resolve() == HD1910_PARAMS.resolve(), name
                assert actuator.kp_fw == 5.0 and actuator.vin_min == 4.0, name
            spec_fn = robot.spec_fn
            if spec_fn not in cache:
                spec = spec_fn()
                # Each mesh must resolve to the selected local STL directory.
                for mesh in spec.meshes:
                    path = Path(mesh.file)
                    if not path.is_absolute():
                        path = Path(spec.modelfiledir) / spec.meshdir / path
                    assert path.resolve().is_relative_to((ROBOT_DIR / "assets").resolve()), (name, path)
                model = spec.compile()
                assert model.nu == 14, (name, model.nu)
                cache[spec_fn] = {
                    "massKg": float(model.body_mass.sum()), "actuators": model.nu,
                    "meshCount": model.nmesh,
                }
            assert abs(cfg.sim.mujoco.timestep * cfg.decimation - .02) < 1e-9, name
            rows.append({"task": name, "play": play, **cache[spec_fn]})
    assert rows, "No local tasks registered"
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_inputs()
    if args.configs:
        report["tasks"] = audit_tasks()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
