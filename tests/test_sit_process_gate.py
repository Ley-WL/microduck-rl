import copy,importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('skill_gpu_process',Path(__file__).parents[1]/'scripts/server/evaluate_skills.py')
gpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(gpu)


def result(travel=20,step=5):
    return [{'scenario':'stand_cycle','seed':42,'envs':1,'initialStateDigest':'paired','successByTrial':[True],
             'transitionSmoothnessByTrial':[[{'time':2.,'excessTravelDeg':[travel,travel],'maxFrameChangeDeg':[step,step]}]]}]


def test_endpoint_success_does_not_hide_shaking_or_missing_process_reference():
    assert not gpu.compare_process_results(result(),None)['passed']
    assert gpu.compare_process_results(result(),result())['passed']
    assert not gpu.compare_process_results(result(40),result())['passed']
    assert not gpu.compare_process_results(result(step=10),result())['passed']


def test_smoothness_cannot_pass_by_losing_a_successful_motion_or_reference_trial():
    candidate=result();candidate[0]['successByTrial']=[False]
    assert not gpu.compare_process_results(candidate,result())['passed']
    candidate=result();candidate[0]['seed']=43
    assert not gpu.compare_process_results(candidate,result())['passed']
    candidate=result();candidate[0].pop('transitionSmoothnessByTrial')
    assert not gpu.compare_process_results(candidate,result())['passed']


def test_process_reference_requires_complete_trials_and_matching_windows():
    candidate=result();candidate[0]['successByTrial']=[]
    assert not gpu.compare_process_results(candidate,result())['passed']
    candidate=result();candidate[0]['profile']='robust'
    assert not gpu.compare_process_results(candidate,result())['passed']
    candidate=result();candidate[0]['transitionSmoothnessByTrial'][0][0]['time']=3.
    assert not gpu.compare_process_results(candidate,result())['passed']
    candidate=result();candidate[0]['initialStateDigest']='different'
    assert not gpu.compare_process_results(candidate,result())['passed']
