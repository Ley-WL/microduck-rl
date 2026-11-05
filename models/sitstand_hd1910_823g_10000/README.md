# 823 g 坐下站起 · 10,000 轮

用户观看最终仿真后认可本候选，要求提交保存。推理文件为 `policy.onnx`，完整 PPO 检查点为 `model_9999.pt`，实际完成 10,000 次更新。

## 训练方式

使用官方 SitStand 配方，从随机初始化开始，4096 个并行环境，seed 211121。采用 HD1910 执行器、Observer 机械与动作目标限位，配重通过 `walk_mass_823g.apply_mass_823g` 应用到 `sitstand_observer_official.spec`，编译总质量为 0.823 kg。

训练入口为 `scripts/server/train_sitstand_official.py`，本次运行预算覆盖为 10,000 次更新。网络、PPO、课程见 `training-contract.json`，实际完成计数见 `training-complete.json`，运行源码哈希见 `source-manifest.json`。完整运行源码和中间检查点保留于服务器 sitstand823-10000-20261105 目录。

## 最终评估

823 g、7.4 V 名义仿真，四种场景各 16 个环境。每段指令 6 秒，连续四段；各段最后 2 秒至少 95% 的帧需满足姿态与稳定性判据。

| 工况 | 达标 | 未倒 |
|---|---:|---:|
| 站姿起步，交替坐下站起 | 16/16 | 16/16 |
| 坐姿起步，交替站起坐下 | 15/16 | 16/16 |
| 保持站立 | 16/16 | 16/16 |
| 保持坐姿 | 16/16 | 16/16 |
| 合计 | 63/64 | 64/64 |

旧版同条件为 64/64 达标。逐例数据见 `results.json`、`baseline-results.json`，训练阶段对照见 `comparison.json`。实物效果待测。

| 文件 | SHA-256 |
|---|---|
| policy.onnx | `1f72ffdf18e67f5895bba4a10d9567b76e69718104d108ab78d1b8187f87bdca` |
| model_9999.pt | `b9c97c40a40a6ffa29a2febd3873bf68f7946ba2e10cc4d719cf3262f7db2be7` |
