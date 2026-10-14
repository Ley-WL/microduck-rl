"""All default paths must agree on the local hardware, including old task IDs."""
import importlib.util
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_hardware_inputs_and_all_registered_tasks():
    spec = importlib.util.spec_from_file_location("audit_hardware", ROOT / "scripts/audit_hardware_profile.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    assert audit.audit_inputs()["verifiedFiles"] > 20
    assert len(audit.audit_tasks()) > 20


def test_local_mjcf_has_no_legacy_pd_and_cleanup_is_idempotent():
    import mujoco
    import numpy as np
    from mjlab_microduck.robot.hardware_profile import ROBOT_DIR
    from mjlab_microduck.robot.strip_legacy_actuators import clean_xml
    for path in ROBOT_DIR.glob("robot_*.xml"):
        text = path.read_text(encoding="utf-8")
        assert clean_xml(text) == text
        assert 'class="chosen_actuator_old"' not in text
        model = mujoco.MjModel.from_xml_path(str(path))
        assert model.nu == 14
        np.testing.assert_array_equal(model.actuator_gainprm[:, 0], 0.)


def test_new_exports_identify_hd1910():
    from mjlab_microduck.publish.manifest import ROBOT
    from mjlab_microduck.robot.hardware_profile import provenance
    assert ROBOT["servos"] == "hd1910"
    source = provenance()
    assert len(source["actuatorParamsSha256"]) == 64
    assert source["hardwareValidated"] is False


def test_backlash_feetech_resets_only_selected_worlds():
    from types import SimpleNamespace
    from unittest.mock import patch
    from mjlab_microduck.actuator.feetech_bam import FeetechBacklashActuator
    from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator

    actuator = object.__new__(FeetechBacklashActuator)
    actuator._goal_reset_pending = torch.tensor([[False], [False]])
    actuator._bam_model = SimpleNamespace(actuator=SimpleNamespace(
        q_target_smooth=torch.tensor([[1., 2.], [3., 4.]])))
    with patch.object(FrictionDRBamActuator, "reset"):
        actuator.reset(torch.tensor([1]))
    captured = []
    def compute(self, cmd):
        captured.append(self._bam_model.actuator.q_target_smooth.clone())
        return torch.zeros_like(cmd.pos)
    with patch.object(FrictionDRBamActuator, "compute", compute):
        actuator.compute(SimpleNamespace(pos=torch.tensor([[9., 9.], [5., 6.]])))
    torch.testing.assert_close(captured[0], torch.tensor([[1., 2.], [5., 6.]]))
    assert not actuator._goal_reset_pending.any()
