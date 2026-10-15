"""Independent NumPy BAM oracle against our Torch friction path; no GPU needed."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from bam.model import load_model
import mjlab.envs  # Complete task discovery before importing the actuator module.
from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator

root = Path(__file__).resolve().parents[2]
bam = load_model(str(root/'src/mjlab_microduck/robot/hd1910/1910_m6.json'))
axis = np.linspace(-1., 1., 21)
motor, external, velocity = np.meshgrid(axis, axis, [-3., -.2, 0., .2, 3.], indexing='ij')
for dtype, tolerance in [(torch.float64, 1e-12), (torch.float32, 1e-7)]:
    # Compare identical representable inputs. Float32 can collapse nearly equal
    # magnitudes to exact equality, where the reference intentionally gates off.
    motor_t = torch.tensor(motor, dtype=dtype)
    external_t = torch.tensor(external, dtype=dtype)
    velocity_t = torch.tensor(velocity, dtype=dtype)
    expected, _ = bam.compute_frictions(motor_t.double().numpy(),
        external_t.double().numpy(), velocity_t.double().numpy())
    coefficient = torch.exp(-(velocity_t.abs() / bam.dtheta_stribeck.value) ** bam.alpha.value)
    for scale in (.9, 1., 1.1):
        obj = SimpleNamespace(_bam_model=bam, friction_scale=scale)
        actual = FrictionDRBamActuator._compute_friction_budget(obj,
            motor_t, external_t, coefficient).numpy()
        np.testing.assert_allclose(actual, expected*scale, atol=tolerance, rtol=tolerance)
print('BAM friction parity passed: 2205 sign/load/speed cases, 2 dtypes, 3 scales')
