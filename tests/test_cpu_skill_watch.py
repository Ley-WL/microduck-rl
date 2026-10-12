import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'cpu_watch', Path(__file__).parents[1] / 'scripts/server/watch_skills_cpu.py')
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def test_checkpoint_scan_ignores_probe_logs_and_unrelated_directories(tmp_path):
    checkpoint = tmp_path / 'checkpoint-1000'
    checkpoint.mkdir()
    (tmp_path / 'checkpoint-1000-probe.log').write_text('evaluation log')
    (tmp_path / 'checkpoint-2000').write_text('not a directory')
    (tmp_path / 'checkpoint-old').mkdir()
    assert watch.checkpoint_directories(tmp_path) == [checkpoint]


def test_headless_cpu_video_environment_does_not_inherit_glfw(monkeypatch, tmp_path):
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.setenv('MUJOCO_GL', 'glfw')
    monkeypatch.setenv('PYTHONPATH', '/stale/source')
    for kind, gpu in [('getup', '4'), ('sitstand', '6')]:
        environment = watch.cpu_runtime_environment(tmp_path, kind)
        assert environment['MUJOCO_GL'] == 'egl'
        assert environment['MUJOCO_EGL_DEVICE_ID'] == gpu
        assert environment['PYTHONPATH'] == str(tmp_path/'src')
        assert environment['OMP_NUM_THREADS'] == '2'
