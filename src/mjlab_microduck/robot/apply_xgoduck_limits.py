"""Keep CAD exports aligned with the reviewed XgoDuck 14-joint ranges."""

import json
from pathlib import Path
import re
import sys


def apply_limits(path: Path) -> None:
    limits = json.loads(Path(__file__).with_name("xgoduck_joint_limits.json").read_text())["joints"]
    found = set()

    def replace(match):
        tag = match.group().decode("utf-8")
        name = re.search(r'\bname="([^"]+)"', tag)
        if name and name[1] in limits:
            joint = name[1]
            if joint in found:
                raise ValueError(f"Duplicate servo joint: {joint}")
            found.add(joint)
            value = " ".join(str(x) for x in limits[joint]["range"])
            tag, count = re.subn(r'\brange="[^"]+"', f'range="{value}"', tag)
            if count != 1:
                raise ValueError(f"Missing range: {joint}")
        return tag.encode("utf-8")

    updated = re.sub(rb'<joint\b[^>]*>', replace, path.read_bytes())
    if found != set(limits):
        raise ValueError(f"Missing servo joints: {set(limits) - found}")
    path.write_bytes(updated)


if __name__ == "__main__":
    apply_limits(Path(sys.argv[1]))
