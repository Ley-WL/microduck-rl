"""Mass and inertia adjustments shared by 823 g HD1910 training and evaluation."""

SERVO_COUNTS = {
    "trunk_base": 2, "yaw2roll": 1, "upper_leg_left": 2, "leg": 1, "neck": 2,
    "yaw_roll_motion": 1, "jaw_soft": 2, "bearing_roll": 1, "upper_leg_right": 2, "leg_2": 1,
}
EXTRA_KG = {"trunk_base": 0.00475682, "jaw_soft": 0.006}


def apply_mass_823g(spec):
    original = spec.compile()
    for name, count in SERVO_COUNTS.items():
        body = spec.body(name)
        index = original.body(name).id
        mass = original.body_mass[index]
        body.mass = mass + count * 0.005 + EXTRA_KG.get(name, 0.0)
        body.fullinertia = [float("nan"), 0, 0, 0, 0, 0]
        body.iquat = original.body_iquat[index]
        body.ipos = original.body_ipos[index]
        body.inertia = original.body_inertia[index] * (body.mass / mass)
    assert abs(spec.compile().body_mass.sum() - 0.823) < 1e-9
    return spec
