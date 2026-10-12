"""Observe owned training, export snapshots, probe, then run held-out validation.

Never restarts a job because a status file is absent; checks its actual PID.
Never equates a 64-world probe or a training reward with 95% acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--kind', choices=['getup', 'sitstand'], required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--iterations', type=int, required=True)
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.run):
        raise ValueError('Invalid run ID')
    root = Path(__file__).resolve().parents[2]
    runs = root.parent / 'runs'
    output = root.parent / 'exports/skills-20261009' / a.run
    output.mkdir(parents=True, exist_ok=True)
    state = runs / (a.run + '.watch-status.json')
    pid = int((runs / (a.run + '.pid')).read_text())
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(root/'src'), CUDA_VISIBLE_DEVICES=a.gpu, MUJOCO_GL='egl', OMP_NUM_THREADS='4')
    task = f"Mjlab-{'Getup' if a.kind == 'getup' else 'SitStand'}-Flat-MicroDuck-HD1910"
    experiment = f'hd1910_{a.kind}_v1'
    processed = set()

    def report(stage, **details):
        payload = {'run': a.run, 'kind': a.kind, 'stage': stage, 'time': time.time(), **details}
        state.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload), flush=True)

    def execute(command, log_name):
        with (output/log_name).open('w') as log:
            subprocess.run([sys.executable, *command], cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)

    def evaluate(policy, checkpoint_dir, profile, envs, seeds, label, video=False):
        destination = checkpoint_dir / label
        command = ['scripts/server/evaluate_skills.py', '--kind', a.kind, '--onnx', str(policy), '--output', str(destination),
                   '--profile', profile, '--envs', str(envs), '--seeds', *map(str, seeds)]
        if video:
            command.append('--video')
        execute(command, f'{checkpoint_dir.name}-{label}.log')
        return json.loads((destination/'summary.json').read_text())

    report('waiting_for_checkpoint', trainingPid=pid)
    while True:
        exit_file = runs / (a.run + '.exit')
        terminal = exit_file.exists()
        if not terminal:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                time.sleep(15)
                if not exit_file.exists():
                    report('training_process_missing', trainingPid=pid, accepted=False)
                    return
                terminal = True
        if terminal and int(exit_file.read_text()) != 0:
            report('training_failed', exitCode=int(exit_file.read_text()), accepted=False)
            return
        directories = list((root/'logs/rsl_rl'/experiment).glob('*_'+a.run))
        if len(directories) > 1:
            raise RuntimeError('Ambiguous run directory')
        checkpoints = []
        if directories:
            for path in directories[0].glob('model_*.pt'):
                iteration = int(path.stem.split('_')[-1])
                if iteration not in processed and path.stat().st_size > 0 and time.time()-path.stat().st_mtime > 10 and (iteration >= 1000 or terminal):
                    checkpoints.append((iteration, path))
        if checkpoints:
            iteration, checkpoint = max(checkpoints)
            if not terminal and processed and iteration - max(processed) < 1000:
                time.sleep(30)
                continue
            destination = output / f'checkpoint-{iteration}'
            destination.mkdir(exist_ok=True)
            policy = destination / (a.kind+'.onnx')
            report('exporting', checkpoint=str(checkpoint), trainingPid=pid)
            execute(['scripts/export.py', task, '--checkpoint-file', str(checkpoint), '--onnx-file', str(policy), '--num-envs', '1'], f'{iteration}-export.log')
            execute(['scripts/server/verify_onnx.py', str(policy)], f'{iteration}-contract.log')
            execute(['scripts/server/write_policy_metadata.py', str(policy), task], f'{iteration}-metadata.log')
            report('probing', checkpoint=str(checkpoint))
            probe = evaluate(policy, destination, 'nominal', 64, [31415], 'probe')
            full_candidate = all(result['successRate'] >= .95 for result in probe['results'])
            accepted = False
            if full_candidate or terminal:
                report('held_out_validation', checkpoint=str(checkpoint))
                nominal = evaluate(policy, destination, 'nominal', 200, [1009, 2027], 'nominal')
                robust = evaluate(policy, destination, 'robust', 200, [1009, 2027], 'robust')
                accepted = nominal['accepted'] and robust['accepted']
                if accepted:
                    evaluate(policy, destination, 'nominal', 8, [1009], 'video', video=True)
            processed.add(iteration)
            report('checkpoint_evaluated', checkpoint=str(checkpoint), probe=probe['results'], accepted=accepted,
                   artifacts=str(destination), hardwareValidated=False, videoReviewRequired=accepted)
        if terminal and (a.iterations-1) in processed:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
