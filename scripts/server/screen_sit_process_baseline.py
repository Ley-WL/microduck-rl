"""Cache the unchanged deployed seed under the v2 evaluation conditions."""
import json,shutil
import finalize_sit_transfer as f
f.task='Mjlab-SitStandProcess-Flat-MicroDuck-HD1910'
f.extra_conditions={'gain-low':['--actuator-kp-scale','.8'],'gain-high':['--actuator-kp-scale','1.2']}
baseline=f.evidence/'baseline.onnx'
shutil.copy2(f.root/'weight-1p0.onnx',baseline)
metadata=json.loads((f.root/'weight-1p0.metadata.json').read_text());metadata['task']=f.task
baseline.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2))
reports=f.screen(baseline,'baseline')
result={key:f.summarize(value)[1] for key,value in reports.items()}
(f.evidence/'baseline-preview.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
