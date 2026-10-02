# 服务器训练

2026-10-01：训练工作目录迁到服务器，正式训练在服务器执行。本地保留代码维护与文档。SSH 密码不存入仓库。

## 环境与目录

- SSH：`leiwu30@8.133.210.70`，端口 `8422`。
- 系统：Ubuntu 24.04 x86_64，8张 NVIDIA RTX 4090；显存以 `nvidia-smi` 为准。
- 项目：`/home/leiwu30/microduck-training/microduck-rl`。
- 环境：项目独立 `.venv`，Python 3.12，`uv sync --locked`。
- 源码基线：官方 `pollen-robotics/microduck_rl`，`develop`，`cfe1c2adcceb55f6b6e369c888b31c6873175c55`。服务器由本地 Git bundle 克隆，保留历史。
- 启动日志、PID和完成状态：`/home/leiwu30/microduck-training/runs/`。
- 训练断点、TensorBoard指标和配置：项目 `logs/rsl_rl/<experiment_name>/`。
- ONNX输出：`/home/leiwu30/microduck-training/exports/`。

本机使用专用 SSH 密钥 `C:/Users/wl/.ssh/microduck_training_ed25519`，独立主机记录 `C:/Users/wl/.ssh/microduck_training_known_hosts`；私钥与密码均不上传到训练目录。

```powershell
ssh -p 8422 -i C:/Users/wl/.ssh/microduck_training_ed25519 -o UserKnownHostsFile=C:/Users/wl/.ssh/microduck_training_known_hosts leiwu30@8.133.210.70
```

## 短测试与正式训练

脚本默认仅使用物理GPU7；登录后先检查占用，必要时通过 `MICRODUCK_GPU` 选择其他空闲卡。`CUDA_VISIBLE_DEVICES` 映射后，程序中的 `cuda:0` 对应所选物理卡。默认 TensorBoard 本地日志，无需 W&B 登录。

```bash
cd ~/microduck-training/microduck-rl
nvidia-smi

# 每个新配置先64环境、5迭代；这不是有效训练成品。
MICRODUCK_GPU=7 scripts/server/smoke.sh Mjlab-StandUp-Flat-MicroDuck

# 正式训练示例，按后续任务决定是否启动。
MICRODUCK_GPU=7 scripts/server/launch.sh standup-001 Mjlab-StandUp-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max-iterations 1000 --agent.run-name standup-001

tail -f ../runs/standup-001.log
cat ../runs/standup-001.exit  # 完成后出现，0为正常退出
```

断点续训与跨任务暖启动不同，后者须遵守 AGENTS.md 的课程与归一化要求。

```bash
MICRODUCK_GPU=7 scripts/server/launch.sh standup-resume-001 Mjlab-StandUp-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max-iterations 1500 \
  --agent.resume True --agent.load-checkpoint model_999.pt
```

以上续训按上游最新运行目录选择机制查找断点；执行前核对帮助与所选运行，避免误接其他任务的断点。

## 导出

```bash
scripts/server/export.sh Mjlab-StandUp-Flat-MicroDuck \
  /absolute/path/to/model_999.pt ../exports/standup-001.onnx
~/.local/bin/uv run --locked python scripts/server/verify_onnx.py ../exports/standup-001.onnx
```

使用上游导出器，包含观测归一化；导出后再检查输入61维/输出14维和有限数值，并做仿真回放。短测试模型不能直接用于实机。

## 验证边界

官方仿真执行器仍是XL330 BAM模型，尚未适配本机HD1910。环境安装、短训练和导出成功仅证明服务器训练链路可用，不证明站立、起身或实机迁移成功。不连接主控、不上传策略到机器人、不自动启动长期训练。

## 2026-10-01 部署验收

- 锁定依赖安装完成，Python3.12、PyTorch2.9.1+cu128，CUDA可用。
- GPU7运行StandUp任务64环境、5次迭代正常结束，nan_state计数均0，保存model_0.pt与model_4.pt。
- 上游导出器成功导出setup-smoke.onnx，CPU ONNX Runtime核验输入[1,61]、输出[1,14]，零观测和随机小量观测输出均为有限值。
- 后台启动器通过帮助命令验收，PID/日志/退出状态文件生成，退出状态0；未启动长期训练。
- 服务器setup/deployment-verification.json保存源码基线、脚本哈希、导出哈希与检查结果。
