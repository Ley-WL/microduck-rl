"""Read-only CAD collision audit; no automatic exclusions or robot commands."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run():
    spec = mujoco.MjSpec.from_file(str(ROOT/'src/mjlab_microduck/robot/replica/robot_walk.xml'))
    # Inspect convex mesh contacts before choosing a training collision set.
    for index, geom in enumerate(spec.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            geom.name = f'audit_{index}_{geom.meshname}'
            geom.contype, geom.conaffinity = 1, 1
    model = spec.compile()
    data = mujoco.MjData(model)
    metadata = json.loads((ROOT.parent/'microduck-observer/models/alpha_sitstand.metadata.json').read_text())
    home = np.array(metadata['homeRadians'])
    sit = home.copy()
    for index, value in metadata['sitJointOverrides'].items():
        sit[int(index)] = value
    rows = []
    for alpha in np.linspace(0, 1, 11):
        data.qpos[7:] = (1-alpha)*home + alpha*sit
        mujoco.mj_forward(model, data)
        contacts = []
        for contact in data.contact:
            if contact.dist >= 0:
                continue
            geoms = [model.geom(int(g)) for g in contact.geom]
            contacts.append({'geoms': [g.name for g in geoms],
                             'bodies': [model.body(int(model.geom_bodyid[int(g.id)])).name for g in geoms],
                             'penetrationM': float(-contact.dist)})
        rows.append({'sitBlend': float(alpha), 'contacts': contacts})
    return {'hardwareValidated': False, 'note': 'Mesh convex hulls may include empty cavities; overlaps require review.', 'poses': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run(), indent=2), encoding='utf-8')
