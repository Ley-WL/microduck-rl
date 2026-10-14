"""The local robot's hardware inputs; safe to import without mjlab/CUDA."""
from pathlib import Path
import hashlib
import json

ROBOT_ROOT = Path(__file__).resolve().parent
ROBOT_DIR = ROBOT_ROOT / "replica"
HD1910_PARAMS = ROBOT_ROOT / "hd1910" / "1910_m6.json"
HARDWARE_PROFILE = "microduck-replica-hd1910-v1"
KP_FW = 5.0
VIN_RANGE = (7.4, 8.0)
VIN_DROP_GAIN_RANGE = (0.0, 0.2)
VIN_MIN = 4.0
CONTROL_HZ = 50


def provenance():
    """Identify the source snapshot; this does not claim physical validation."""
    verify_inputs()
    return {
        "profile": HARDWARE_PROFILE,
        "actuatorParamsSha256": hashlib.sha256(HD1910_PARAMS.read_bytes()).hexdigest(),
        "geometryManifestSha256": hashlib.sha256((ROBOT_ROOT / "hardware-inputs.json").read_bytes()).hexdigest(),
        "hardwareValidated": False,
    }


def verify_inputs():
    manifest = json.loads((ROBOT_ROOT / "hardware-inputs.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((ROBOT_ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Hardware input changed; review and update manifest: {name}")
    return len(manifest["sha256"])


def actuator_kwargs():
    """Return independent kwargs; task-specific DR may override explicitly."""
    return dict(
        json_path=str(HD1910_PARAMS),
        target_names_expr=(r"^(?!passive_).*",),
        kp_fw=KP_FW,
        vin_range=VIN_RANGE,
        vin_drop_gain_range=VIN_DROP_GAIN_RANGE,
        vin_min=VIN_MIN,
        delay_min_lag=3,
        delay_max_lag=6,
    )
