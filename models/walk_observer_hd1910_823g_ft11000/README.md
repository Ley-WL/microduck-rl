# HD1910 · 823 g 行走模型

从 Observer 5000 轮行走策略出发，在 823 g 配重下微调 1,000 + 10,000 轮。推理用 `policy.onnx`，继续 PPO 训练可加载完整检查点 `model_10999.pt`。

## 训练

在仓库根目录，使用项目的 Linux CUDA 训练环境执行。两个阶段使用同一个新输出目录。

```sh
export PYTHONPATH=src MUJOCO_GL=egl
python scripts/server/train_walk_823g.py warmup --runs logs/walk823
python scripts/server/train_walk_823g.py continue --runs logs/walk823
```

训练沿用官方平地速度任务、Observer 机械限位、动作目标限位和 HD1910 执行器。15 个舵机各加 5 g，躯干额外加 4.75682 g，头部额外加 6 g。按各部件质量比缩放惯量，保留惯性坐标系。

每阶段使用 4096 个并行环境，每轮 24 步。第一阶段从仓库中的 5000 轮 ONNX 恢复 actor 和归一化参数，归一化计数设为 1,000,000，探索标准差设为 0.10，重新初始化 critic 和优化器。第二阶段完整加载第一阶段检查点，包括优化器和课程步数。两个阶段均用 128 组输入核对起始 actor 和对应 ONNX，最大绝对误差要求小于 2e-5。

## 回放已发布模型

```sh
export PYTHONPATH=src MUJOCO_GL=egl WALK_PROFILE=823g
export WALK_POLICY_PATH=models/walk_observer_hd1910_823g_ft11000/policy.onnx
WALK_OUTPUT_DIR=evidence/walk823-final python scripts/compare_walk_inputs.py
```

`results.json` 是本次训练后的逐工况结果。31 组指令，每组 16 个环境。共 496/496 未跌倒，运动工况 480/480 达到停步判据。原地转向指令 ±0.75 rad/s 时实测左 0.685、右 −0.731 rad/s；±0.2 rad/s 时几乎无响应。这是 7.4 V、14 ms 执行器延迟下的仿真结果，实机待测。

## SHA-256

| 文件 | SHA-256 |
|---|---|
| policy.onnx | `21a3afdf6a120ad910227b5023e16456c263e1c73e38275ad52aca0e1bb8000f` |
| model_10999.pt | `301c7fd4e9be7283741b5ce22c9040ddffb121ff014f4382c81e89a9086b37f0` |
