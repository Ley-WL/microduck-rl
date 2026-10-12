"""CPU deployment rehearsal for independently GPU-validated skill candidates."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def checkpoint_directories(export_root):
    # Probe logs share the checkpoint prefix but are ordinary files.
    return sorted(path for path in export_root.glob('checkpoint-*')
                  if path.is_dir() and re.fullmatch(r'checkpoint-\d+', path.name))


def cpu_runtime_environment(root, kind):
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='2', PYTHONPATH=str(root/'src'),
                       MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='4' if kind == 'getup' else '6')
    return environment


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--kind', choices=['getup', 'sitstand'], required=True)
    p.add_argument('--detach', action='store_true')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.run):
        raise ValueError('Invalid run name')
    root = Path(__file__).resolve().parents[2]
    runs = root.parent/'runs'
    status = runs/(a.run+'.cpu-watch-status.json')
    pid_file = runs/(a.run+'.cpu-watch.pid')
    if a.detach:
        if pid_file.exists():
            raise RuntimeError('CPU watcher PID file exists; inspect the process before retrying')
        with (runs/(a.run+'-cpu-watch.log')).open('a') as log:
            process = subprocess.Popen([sys.executable, __file__, '--run', a.run, '--kind', a.kind],
                                       cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pid_file.write_text(str(process.pid)+'\n')
        print('CPU watcher PID:', process.pid)
        return
    export_root = root.parent/'exports/skills-20261009'/a.run
    gpu_watch_pid = int((runs/(a.run+'.watch.pid')).read_text())
    processed = set()
    environment = cpu_runtime_environment(root, a.kind)

    def report(stage, **fields):
        value = {'run': a.run, 'kind': a.kind, 'stage': stage, 'time': time.time(), **fields}
        status.write_text(json.dumps(value, indent=2))
        print(json.dumps(value), flush=True)

    report('waiting_for_gpu_validated_candidate', gpuWatcherPid=gpu_watch_pid)
    while True:
        for checkpoint in checkpoint_directories(export_root):
            if checkpoint.name in processed:
                continue
            try:
                nominal = json.loads((checkpoint/'nominal/summary.json').read_text())
                robust = json.loads((checkpoint/'robust/summary.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if not nominal.get('accepted') or not robust.get('accepted'):
                continue
            policy = checkpoint/(a.kind+'.onnx')
            policy_hash = nominal['policySha256']
            if policy_hash != robust['policySha256']:
                raise RuntimeError('GPU validation policy hashes differ')
            report('cpu_runtime_rehearsal', checkpoint=str(checkpoint), policySha256=policy_hash)
            destination = checkpoint/'cpu-runtime'
            try:
                with (checkpoint/'cpu-runtime.log').open('w') as log:
                    subprocess.run([sys.executable, 'scripts/evaluate_skills_cpu.py', '--onnx', str(policy),
                                    '--output', str(destination), '--trials', '200', '--seeds', '1009', '2027', '--video'],
                                   cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                report('cpu_runtime_failed', checkpoint=str(checkpoint), policySha256=policy_hash,
                       exitCode=error.returncode, artifacts=str(destination), accepted=False)
                raise
            summary = json.loads((destination/'summary.json').read_text())
            if summary['policySha256'] != policy_hash:
                raise RuntimeError('CPU validation policy hash differs')
            processed.add(checkpoint.name)
            report('cpu_runtime_evaluated', checkpoint=str(checkpoint), accepted=summary['accepted'],
                   groups=[{key: group[key] for key in ('scenario', 'seed', 'successRate')} for group in summary['groups']],
                   artifacts=str(destination), hardwareValidated=False, videoReviewRequired=summary['accepted'])
        # A state file is not proof of a live worker. Check the real PID.
        try:
            os.kill(gpu_watch_pid, 0)
        except ProcessLookupError:
            # Re-scan once after its final writes have settled.
            if getattr(main, '_gpu_terminal_seen', False):
                report('gpu_watcher_finished', processedCheckpoints=sorted(processed), goalCompletionRequiresArtifactReview=True)
                return
            main._gpu_terminal_seen = True
        time.sleep(30)


if __name__ == '__main__':
    main()
