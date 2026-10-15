import importlib.util
import json
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('dance_analysis',Path(__file__).parents[1]/'scripts/analyze_dance10.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_standing_still_and_reset_after_fall_never_pass_as_dance(tmp_path):
    t=np.linspace(0,16,801)
    target=np.zeros((len(t),3,4))
    target[:,:,2]=.25*np.sin(t[:,None]*5)
    target[:,:,1]=.12*np.sin(t[:,None]*5)
    actual=target.copy();actual[:,0]=0
    (tmp_path/'summary.json').write_text(json.dumps({'firstFailureSeconds':[-1,4,-1]}))
    np.savez(tmp_path/'trace.npz',time=t,head=actual,headCommand=target)
    report=module.motion_metrics(tmp_path)
    assert report['survived']==2
    assert report['danceScreenPassed']==1
