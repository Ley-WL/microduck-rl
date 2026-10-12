import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg


def test_ablation_changes_are_isolated_and_preserve_motion_contract():
    for case in ('Coverage','WeakProcess','Temporal','Sensor'):
        task='Mjlab-SitStandProcess'+case+'-Flat-MicroDuck-HD1910'
        cfg=load_env_cfg(task);rl=load_rl_cfg(task)
        assert cfg.actions['joint_pos'].alpha==.45 and cfg.actions['joint_pos'].clip
        assert cfg.events['transfer_pitch_kick'].params['probability']==1.
        assert cfg.rewards['target_acceleration'].weight==-.05
        assert rl.algorithm.onpolicy_retention_coef==2.
        assert rl.algorithm.temporal_coef==(3. if case=='Temporal' else .3)
        assert ('knee_process' in cfg.rewards)==(case in ('WeakProcess','Sensor'))
        if 'knee_process' in cfg.rewards:assert cfg.rewards['knee_process'].weight==-.015
        assert cfg.observations['actor'].terms['base_ang_vel'].noise.n_max==(.5 if case=='Sensor' else .06)
