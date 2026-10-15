"""Audit paired CPU rollouts and direction-specific GPU pulse-student results."""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(root):
    report = {"accepted": False, "simulationOnly": True, "cpu": {}, "gpu": {}}
    reference = root / "baseline" / "cpu-straight"
    for stage in ("baseline", "step100", "step500"):
        rows = json.loads((root / stage / "cpu-body-tracking.json").read_text())["results"]
        paired = 0
        for path in sorted((root / stage / "cpu-straight").glob("seed*.json")):
            actual = json.loads(path.read_text())["frames"]
            baseline = json.loads((reference / path.name).read_text())["frames"]
            assert actual[0]["qpos"] == baseline[0]["qpos"], path
            assert actual[0]["velocity"] == baseline[0]["velocity"], path
            assert [f["command"] for f in actual] == [f["command"] for f in baseline], path
            paired += 1
        assert paired == len(rows)
        report["cpu"][stage] = {"paired": paired, "survived": sum(r["completed"] for r in rows), "rows": rows}
    data = json.loads((root / "joint-pulse-screen.json").read_text())
    cases = np.repeat(np.asarray(data["cases"]), 4, axis=0)
    for label in ("baseline", "candidate"):
        rows = [r for r in data["results"] if r["policy"] == label]
        groups = []
        for speed, yaw in data["cases"]:
            ids = np.all(np.isclose(cases, [speed, yaw]), axis=1)
            metrics = np.concatenate([np.asarray(r["metrics"])[ids] for r in rows])
            feet = np.concatenate([np.asarray(r["phaseStepDistance"])[ids] for r in rows])
            counts = np.concatenate([np.asarray(r["phaseStepCount"])[ids] for r in rows])
            live = np.concatenate([np.asarray(r["survival"])[ids] for r in rows])
            groups.append(dict(command=[speed, yaw], survived=int(live.sum()), worlds=len(live),
                # Retain failed worlds; phase 1 reverses both commanded speed and yaw.
                phaseMean=metrics.mean(0).tolist(),
                phaseFootTravelIncludingNoSteps=feet.mean((0, 2)).tolist(),
                phaseFootfalls=counts.sum((0, 2)).tolist(),
                stopMean=np.concatenate([np.asarray(r["stop"])[ids] for r in rows]).mean(0).tolist()))
        report["gpu"][label] = dict(survived=sum(sum(r["survival"]) for r in rows), groups=groups)
    report["gpu"]["pairing"] = []
    for seed in sorted({r["seed"] for r in data["results"]}):
        a, b = [next(r for r in data["results"] if r["seed"] == seed and r["policy"] == label)
                for label in ("baseline", "candidate")]
        assert a["initialStateSha256"] == b["initialStateSha256"]
        matched = [x == y for x, y in zip(a["commandWorldSha256"], b["commandWorldSha256"])]
        both = np.asarray(a["survival"]) & np.asarray(b["survival"])
        assert all(match for match, survives in zip(matched, both) if survives)
        report["gpu"]["pairing"].append(dict(seed=seed, matchedWorlds=sum(matched),
            lost=np.flatnonzero(np.asarray(a["survival"]) & ~np.asarray(b["survival"])).tolist()))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(cpu={k: v["paired"] for k, v in result["cpu"].items()},
        gpu={k: v["survived"] for k, v in result["gpu"].items() if k != "pairing"},
        pairing=result["gpu"]["pairing"])))
