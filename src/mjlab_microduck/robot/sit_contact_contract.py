"""Candidate sit/stand physics; shared by training and CPU rehearsal.

Robot bit 1 contacts terrain; bit 2 enables robot self contact. CAD convex
hulls of the closed jaw overlap the neck at HOME and throughout the sweep;
exclude that assembly pair explicitly, never auto-exclude new collisions.
"""
import json
from pathlib import Path

REVISION = 'sit-self-contact-bounded-target-v3'
LIMITS = json.loads((Path(__file__).parent/'xgoduck_joint_limits.json').read_text())['joints']
TARGET_LIMITS = {name: tuple(entry['range']) for name, entry in LIMITS.items()}
ASSEMBLY_EXCLUDES = (('neck_pitch', 'jaw_soft'),)


def enable_sit_self_contacts(spec):
    for geom in spec.geoms:
        if geom.name.startswith('body_ground_') or geom.name in ('left_foot_collision', 'right_foot_collision'):
            geom.contype, geom.conaffinity = 3, 2
        elif geom.type.name == 'mjGEOM_MESH':
            # No duplicated collision copies of visual meshes.
            geom.contype, geom.conaffinity = 0, 0
    for body1, body2 in ASSEMBLY_EXCLUDES:
        if not any(e.bodyname1 == body1 and e.bodyname2 == body2 for e in spec.excludes):
            spec.add_exclude(bodyname1=body1, bodyname2=body2)
    return spec
