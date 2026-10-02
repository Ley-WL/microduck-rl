"""CPU check of the shared MicroDuck policy interface, not a rollout evaluation."""
import json
import sys

import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
inputs, outputs = session.get_inputs(), session.get_outputs()
assert len(inputs) == len(outputs) == 1, "Expected one observation and one action tensor"
assert inputs[0].shape[-1] == 61, inputs[0].shape
assert outputs[0].shape[-1] == 14, outputs[0].shape
assert inputs[0].type == "tensor(float)", inputs[0].type
rng = np.random.default_rng(0)
for observation in (
    np.zeros((1, 61), dtype=np.float32),
    rng.normal(0, 0.1, (1, 61)).astype(np.float32),
):
    action = session.run(None, {inputs[0].name: observation})[0]
    assert action.shape == (1, 14), action.shape
    assert np.isfinite(action).all(), "Nonfinite policy output"
print(json.dumps({"input": inputs[0].shape, "output": outputs[0].shape,
                  "finiteOutput": True, "provider": "CPUExecutionProvider"}))
