# HD1910 Observer 限位行走：5000 轮

当前采用的最终部署策略；用户于 2026-10-20 和 2026-10-21 反馈效果很好。10000 轮候选没有全面超过本版本，未替换现役模型。

- `policy.onnx`：61 维观测 → 14 维动作，归一化内嵌，50Hz。
- `walk.metadata.json`：关节顺序、HOME、物理范围和运行合同。scale=1，无 EMA，raw previous action，P5/D0。
- SHA256：`e3b90ac0e5359f91e14306b7e2143e0d5a1f4cd49cb256bca3fd71835250d895`。
- `training-contract.json`：原始训练预检记录，4096 环境 × 5000 轮（最终 checkpoint 索引 4999）。
- `simulation-comparison.md`：历史 ae59d61a 与本策略的 30 输入对照；表内“新模型”即本版本，“当前”是当时旧基线。

模型与 Observer 仓库 `models/xgoduck_walk.onnx` 完全相同。运行需要相匹配的共享标定、关节顺序和限位，不能仅复制权重到任意控制器。

短时仿真 30×16 共 480/480 未倒；小幅原地左转弱响应、后退偏航仍是已知局限。用户反馈不是所有场景的定量验收，metadata 的 hardwareValidated=false 不因此改写。

训练与评估命令见 [训练说明](../../docs/walk-observer-official-training-20261020.md)。此包包含部署 ONNX，不包含可恢复 Adam/critic 的训练 checkpoint。
