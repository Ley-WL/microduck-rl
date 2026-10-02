"""Keep v3 learned weights/normalization but reset exploration and Adam moments."""
import hashlib
import json
from pathlib import Path
import sys
import torch

source=Path(sys.argv[1]).resolve()
target=Path(sys.argv[2]).resolve()
if source==target or target.exists():
    raise ValueError("Use a new seed path; do not overwrite the source or an existing seed")
state=torch.load(source,map_location="cpu",weights_only=False)
previous=state["actor_state_dict"]["distribution.std_param"].tolist()
state["actor_state_dict"]["distribution.std_param"].fill_(.25)
state["optimizer_state_dict"]["state"]={}
reset_head_roll="--reset-head-roll" in sys.argv[3:]
if reset_head_roll:
    # Only reset the saturated head_roll output; preserve leg recovery weights.
    actor=state["actor_state_dict"]
    candidates=[k for k,v in actor.items() if k.endswith("weight") and v.ndim==2 and v.shape[0]==14]
    if len(candidates)!=1:
        raise ValueError(f"Ambiguous actor output layer: {candidates}")
    output_key=candidates[0]
    actor[output_key][8].zero_()
    actor[output_key.replace("weight","bias")][8]=0.
target.parent.mkdir(parents=True,exist_ok=True)
torch.save(state,target)
proof={"source":str(source),"sourceSha256":hashlib.sha256(source.read_bytes()).hexdigest(),
       "target":str(target),"targetSha256":hashlib.sha256(target.read_bytes()).hexdigest(),
       "previousExplorationStd":previous,"newExplorationStd":.25,
       "optimizerMomentsReset":True,"learnedWeightsAndNormalizationRetained":True,
       "headRollOutputReset":reset_head_roll}
target.with_suffix(".seed.json").write_text(json.dumps(proof,indent=2),encoding="utf-8")
print(json.dumps(proof))
