from dataclasses import asdict
import re
from mjlab_microduck.tasks.sit_process_ablation_env_cfg import make_sit_ablation_env_cfg,make_sit_ablation_rl_cfg

def test_only_algorithm_difference_is_minibatches():
    before=asdict(make_sit_ablation_rl_cfg('WeakProcess'))
    after=asdict(make_sit_ablation_rl_cfg('FullBatch'))
    before.pop('run_name');after.pop('run_name')
    before['algorithm']['num_mini_batches']=1
    assert before==after
    assert after['num_steps_per_env']==24
    assert after['algorithm']['num_learning_epochs']==2

def test_environment_and_play_contracts_are_identical():
    for play in (False,True):
        before=make_sit_ablation_env_cfg('WeakProcess',play)
        after=make_sit_ablation_env_cfg('FullBatch',play)
        # Entity factories are fresh but equivalent closures; ignore only their addresses.
        stable=lambda cfg:re.sub(r'0x[0-9a-fA-F]+','<address>',repr(cfg))
        assert stable(before)==stable(after)
