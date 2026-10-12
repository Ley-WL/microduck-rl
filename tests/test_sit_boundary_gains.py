import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab_microduck.tasks.sit_boundary_gains import sample_boundary_kp


def test_boundary_buckets_really_cover_the_failed_gain_frontier():
    torch.manual_seed(9017)
    values,bucket=sample_boundary_kp(10000,'cpu')
    assert bool(((values>=.8)&(values<=1.2)).all())
    for i,value in enumerate((.8,.82,1.,1.2)):
        torch.testing.assert_close(values[bucket==i],torch.full_like(values[bucket==i],value),atol=1e-7,rtol=0)
        assert .18<float((bucket==i).float().mean())<.22
    assert bool(((values[bucket==4]>.8)&(values[bucket==4]<1.2)).all())


def test_boundary_case_changes_only_training_gain_distribution():
    a=load_env_cfg('Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910')
    b=load_env_cfg('Mjlab-SitStandProcessBoundaryGain-Flat-MicroDuck-HD1910')
    for name in ('actions','observations','commands','rewards','curriculum','terminations'):
        assert getattr(a,name)==getattr(b,name)
    for name,term in a.events.items():
        if name=='transfer_motor_gains':
            assert term.params==b.events[name].params and term.func!=b.events[name].func
        else:assert term==b.events[name]
    assert load_rl_cfg('Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910').algorithm==load_rl_cfg('Mjlab-SitStandProcessBoundaryGain-Flat-MicroDuck-HD1910').algorithm
