"""Start only after fixed-delay arithmetic preparation, from identical seed state."""
from pathlib import Path
import shutil
import hashlib
import json
import subprocess
import sys
import os
import torch

base = Path('/home/leiwu30/microduck-training')
prep = base / 'walk-delay16-prepare-20261013'
old = base / 'walk-encoder-transfer-20261013'
target = base / 'walk-delay16-transfer-20261013'
assert not target.exists(), 'Track existing case/PID; never duplicate training'
assert (prep / 'evidence/prepare.exit').read_text().strip() == '0'
assert json.loads((prep / 'evidence/prepare-source-check.json').read_text())['passed']
delay = json.loads((prep / 'evidence/delay-implementation-audit.json').read_text())
assert delay['passed'] == 5
target.mkdir()
for folder in ('src','scripts','inputs'):
    shutil.copytree(prep / folder,target / folder)
(target / 'evidence').mkdir()
audit = json.loads((prep / 'evidence/prepare-audit.json').read_text())
# Reuse the original encoder20 seed, including its critic; do not regenerate it
# after a delay-sensitive preparation trajectory and call it matched initialization.
seed = target / 'inputs/seed.pt'
assert seed.read_bytes() == (old / 'inputs/seed.pt').read_bytes()
state = torch.load(seed,map_location='cpu',weights_only=False)
assert not state['optimizer_state_dict']['state']
audit['seedSha256'] = hashlib.sha256(seed.read_bytes()).hexdigest()
audit['seedProvenance'] = 'byte-identical original encoder20 actor/critic/empty Adam'
audit['preparedSeedUsed'] = False
(target / 'inputs/prepare-audit.json').write_text(json.dumps(audit,indent=2))
shutil.copy2(prep / 'evidence/prepare-source-check.json',target / 'inputs/prepare-source-check.json')
shutil.copy2(prep / 'evidence/delay-implementation-audit.json',target / 'inputs/delay-implementation-audit.json')
shutil.copy2(Path(__file__),target / 'scripts/server/launch_walk_delay_control.py')
p = target / 'scripts/server/launch_walk_surface_training.py'
text = p.read_text().replace("onlyBranchDifference='terrain generator'",
    "onlyBranchDifference='fixed actuator delay 14ms -> 16ms; encoder20 same initial state',commandDelaySteps=8")
p.write_text(text)
env = os.environ.copy()
env.update(PYTHONPATH=str(target / 'src'),MICRODUCK_WALK_SURFACE_INPUTS=str(target / 'inputs'),
           MICRODUCK_WALK_TRAIN_ROOTS='Surface',MICRODUCK_WALK_TRAIN_GPU='1')
subprocess.run([sys.executable,str(p)],env=env,cwd=target,check=True)
