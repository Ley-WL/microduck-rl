"""Remove inherited XL330 XML PD templates after a local CAD export.

These placeholders describe actuator names only. HD1910 BAM installs the real
torque limits, armature, voltage control and friction before simulation starts.
"""
from pathlib import Path
import re
import sys


def clean_xml(text):
    for name in ("perfect_actuator", "chosen_actuator_old", "chosen_actuator_new", "chosen_actuator_antoine"):
        text = re.sub(r'(?m)^ *<default class="' + name + r'">.*?</default>\s*\n', '', text, flags=re.S)
    text = re.sub(
        r'<default class="chosen_actuator">.*?</default>',
        '''<default class="chosen_actuator">
      <!-- Geometry-only placeholder. HD1910 BAM supplies all motor dynamics. -->
      <geom contype="0" conaffinity="0"/>
      <joint damping="0" frictionloss="0" armature="0"/>
      <position kp="0" kv="0" ctrllimited="false" forcelimited="false"/>
    </default>''', text, flags=re.S,
    )
    return text


if __name__ == "__main__":
    for name in sys.argv[1:]:
        path = Path(name)
        path.write_text(clean_xml(path.read_text(encoding="utf-8")), encoding="utf-8")
