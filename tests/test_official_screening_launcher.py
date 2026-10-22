"""Screening changes the run budget, not the published task recipe."""
import runpy
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/server/train_pick_roll_official.py'


def test_pick_default_budget(monkeypatch):
    monkeypatch.setenv('OFFICIAL_SKILL', 'pick')
    monkeypatch.delenv('OFFICIAL_SCREEN_UPDATES', raising=False)
    monkeypatch.delenv('OFFICIAL_SEED', raising=False)
    config = runpy.run_path(str(SCRIPT))
    assert config['TARGET'] == config['OFFICIAL_TARGET'] == 20000
    assert config['SEED'] == 211141


def test_seed_screening(monkeypatch):
    monkeypatch.setenv('OFFICIAL_SKILL', 'pick')
    monkeypatch.setenv('OFFICIAL_SCREEN_UPDATES', '3001')
    monkeypatch.setenv('OFFICIAL_SEED', '221149')
    config = runpy.run_path(str(SCRIPT))
    assert config['TARGET'] == 3001
    assert config['OFFICIAL_TARGET'] == 20000
    assert config['SEED'] == 221149


@pytest.mark.parametrize('budget,seed', [('0', '1'), ('20001', '1'), ('3001', '-1')])
def test_invalid_screening(monkeypatch, budget, seed):
    monkeypatch.setenv('OFFICIAL_SKILL', 'pick')
    monkeypatch.setenv('OFFICIAL_SCREEN_UPDATES', budget)
    monkeypatch.setenv('OFFICIAL_SEED', seed)
    with pytest.raises(ValueError):
        runpy.run_path(str(SCRIPT))
