"""Use the same floor contact construction as HD1910 recovery training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))
from mjlab_microduck.robot.ground_contact import add_body_ground_contacts, lowest_mesh_z
