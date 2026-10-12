"""CPU screen an external actor under the same four paired conditions."""
import argparse,json
from pathlib import Path
import finalize_sit_transfer as f
p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path,required=True);p.add_argument('--tag',required=True);a=p.parse_args()
assert a.tag.replace('-','').isalnum()
f.evidence=f.root/'evidence/ablation'
f.extra_conditions={'gain-low':['--actuator-kp-scale','.8'],'gain-high':['--actuator-kp-scale','1.2']}
reports=f.screen(a.onnx,a.tag);result={k:f.summarize(v)[1] for k,v in reports.items()}
(f.evidence/(a.tag+'-preview.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
