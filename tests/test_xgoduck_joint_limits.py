"""The physical limits must survive variant generation and CAD re-export."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/robot"
LIMITS = json.loads((ROOT / "xgoduck_joint_limits.json").read_text())["joints"]
MODELS = sorted(p for family in ("microduck", "replica") for p in (ROOT / family).glob("robot*.xml"))


@pytest.mark.parametrize("path", MODELS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_all_physical_servo_ranges_match_xgoduck(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    assert len(LIMITS) == 14
    for name, entry in LIMITS.items():
        joint = model.joint(name).id
        assert model.jnt_limited[joint]
        np.testing.assert_allclose(model.jnt_range[joint], entry["range"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("family", ["microduck", "replica"])
def test_cad_exports_reapply_reviewed_ranges(family):
    for path in (ROOT / family).glob("config_mjcf*.json"):
        config = json.loads(path.read_text())
        assert f'python3 ../apply_xgoduck_limits.py {config["output_filename"]}.xml' in config["post_import_commands"]
