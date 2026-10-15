"""Compare frozen old/cycle/relative reward rankings, split by seed and objective."""
import argparse
import json
from pathlib import Path

import numpy as np

VARIANTS = dict(old='meanLegacyWeightedRate', cycle='meanCycleWeightedRate', relative='meanWeightedRate')


def compare(data):
    report = dict(accepted=False, training=False, seeds=[], phases=[], excludedWorlds=[], rankingPairs=[])
    for seed in sorted({r['seed'] for r in data['results']}):
        b, c = [next(r for r in data['results'] if r['seed'] == seed and r['policy'] == label)
                for label in ('baseline', 'candidate')]
        assert b['initialStateSha256'] == c['initialStateSha256']
        bb, cc = b['rewardAudit'], c['rewardAudit']
        assert bb['terms'] == cc['terms']
        scores = [{name: np.asarray(audit[key]).sum(-1) for name, key in VARIANTS.items()}
                  for audit in (bb, cc)]
        counts = {objective: dict(eligible=0, correct={v: 0 for v in VARIANTS}) for objective in ('speed', 'yaw')}
        for world in range(len(b['survival'])):
            if not (b['survival'][world] and c['survival'][world]):
                report['excludedWorlds'].append(dict(seed=seed, world=world,
                    baselineSurvived=b['survival'][world], candidateSurvived=c['survival'][world]))
                continue
            assert b['commandWorldSha256'][world] == c['commandWorldSha256'][world]
            for phase in (1, 4):
                cmd = np.asarray(bb['meanCommand'][world][phase])
                np.testing.assert_allclose(cmd, cc['meanCommand'][world][phase], atol=1e-7)
                if min(bb['samples'][world][phase], cc['samples'][world][phase]) < 299:
                    continue
                errors = [abs(np.asarray(audit['meanActual'][world][phase]) - cmd) for audit in (bb, cc)]
                for objective, target, other, margin, tolerance in [('speed', 0, 2, .01, .05), ('yaw', 2, 0, .05, .01)]:
                    winner = next((i for i in (0, 1) if errors[i][target] + margin < errors[1-i][target]
                                   and errors[i][other] <= errors[1-i][other] + tolerance), None)
                    if winner is None:
                        continue
                    counts[objective]['eligible'] += 1
                    report['rankingPairs'].append(dict(seed=seed, world=world, phase=bb['phases'][phase],
                        objective=objective, command=cmd.tolist(),
                        preferredPolicy=('baseline', 'candidate')[winner],
                        errors={label: error.tolist() for label, error in zip(('baseline', 'candidate'), errors)},
                        scores={label: {variant: float(score[variant][world, phase]) for variant in VARIANTS}
                                for label, score in zip(('baseline', 'candidate'), scores)}))
                    for variant in VARIANTS:
                        counts[objective]['correct'][variant] += int(scores[winner][variant][world, phase]
                                                                    > scores[1-winner][variant][world, phase])
        for objective in counts.values():
            objective['relativeNoRegression'] = objective['eligible'] > 0 and objective['correct']['relative'] >= objective['correct']['old']
        report['seeds'].append(dict(seed=seed, objectives=counts))
        for phase, name in enumerate(bb['phases']):
            keep = np.asarray(b['survival']) & np.asarray(c['survival'])
            report['phases'].append(dict(seed=seed, phase=name, pairedWorlds=int(keep.sum()),
                meanRate={label: {variant: float(score[variant][keep, phase].mean()) for variant in VARIANTS}
                          for label, score in zip(('baseline', 'candidate'), scores)}))
    report['relativeRankingNonRegression'] = all(o['relativeNoRegression'] for row in report['seeds'] for o in row['objectives'].values())
    report['limitations'] = ('Diagnostic ranking only. A speed advantage >.01m/s allows yaw error to worsen by <=.05rad/s; '
        'a yaw advantage >.05rad/s allows speed error to worsen by <=.01m/s. Six sampled windows omit portions '
        'of the full cycle, so their rates cannot establish full discounted-return ordering. No gait acceptance.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.source.read_text()))
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['seeds'], indent=2))
