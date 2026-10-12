"""2026-10-12 current board contract, isolated from historical task ranges.

The references only reproduce the current integer target encoding, never write
calibration. A different board calibration must be re-audited before deployment.
"""
import math
import numpy as np

NAMES = ('left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle',
         'neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw',
         'right_hip_roll','right_hip_pitch','right_knee','right_ankle')
IDS = (20,21,22,23,24,30,31,32,33,10,11,12,13,14)
DEGREES = ((-25,30),(-22,22),(-90,90),(-90,90),(-70,70),(-90,70),
           (-90,90),(-170,170),(-20,20),(-30,25),(-22,22),(-90,90),(-90,90),(-70,70))
LIMITS = {n: tuple(math.radians(v) for v in bounds) for n,bounds in zip(NAMES,DEGREES)}
REFERENCES = (2060,2063,2007,2098,2058,1974,1536,2053,2079,2079,2029,2113,2000,2020)
DIRECTIONS = (-1,) * 14
ENCODER_STEP = math.tau / 4096
REVISION = 'kissqy-mechanical-ema045-rawhistory-encoder4096-v1'

def encode_target(target):
    """Rust donor_target for positive valid encodings (round half away from zero)."""
    bounds=np.asarray(list(LIMITS.values()))
    bounded=np.clip(target,bounds[:,0],bounds[:,1])
    encoded=np.floor(np.asarray(REFERENCES)+bounded/ENCODER_STEP*np.asarray(DIRECTIONS)+.5)
    if np.any((encoded<0)|(encoded>4095)):
        raise ValueError('Target cannot be represented within one encoder turn')
    return (encoded-np.asarray(REFERENCES))*ENCODER_STEP*np.asarray(DIRECTIONS)

def edit_mechanical_spec(spec):
    found=set()
    for joint in spec.joints:
        if joint.name in LIMITS:
            joint.range=np.asarray(LIMITS[joint.name])
            joint.limited=True
            found.add(joint.name)
    assert found==set(NAMES), 'Incomplete mechanical joint mapping'
    return spec
