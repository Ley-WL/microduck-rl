import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def test_only_offline_simulation_distribution_changes():
    before=np.load(ROOT/'original-transfer-data.npz');after=np.load(ROOT/'transfer-data.npz')
    assert set(before.files)==set(after.files)
    n=len(before['simulation']);assert after['simulation'].shape==(n*2,61)
    np.testing.assert_array_equal(before['simulation'],after['simulation'][:n])
    for k in before.files:
        if k!='simulation':np.testing.assert_array_equal(before[k],after[k])

def test_every_added_row_is_from_verified_successful_baseline_pool():
    audit=json.loads((ROOT/'critical-replay-dataset-audit.json').read_text())
    assert audit['criticalUniqueRows']==6400 and len(audit['provenance'])==16
    assert audit['realAndPairIndexesExactlyPreserved'] and audit['mixtureExpectedFraction']==.5
    critical=np.load(ROOT/'critical-observations.npz')['observations'];data=np.load(ROOT/'transfer-data.npz')['simulation']
    assert np.isfinite(data).all()
    allowed={r.tobytes() for r in critical}
    assert all(r.tobytes() in allowed for r in data[len(data)//2:])
