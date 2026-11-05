# 823 g 翻滚 · 10,000 轮

2026-11-05 训练完成，按用户要求归档最终模型。推理文件为 `policy.onnx`，可续训的完整 PPO 检查点为 `model_9999.pt`（零基编号，实际 10,000 次更新）。

## 训练方式

使用官方 Roulade 配方，HD1910 执行器、Observer 机械与动作目标限位，4096 个并行环境，seed 211151，从随机初始化训练。各部件采用仓库 `walk_mass_823g.apply_mass_823g` 的配重和惯量缩放，编译总质量 0.823 kg。训练入口为 `scripts/server/train_pick_roll_official.py`，本次运行将 `pick_roll_observer_official.roll_spec` 的返回值应用该配重函数，设置 `OFFICIAL_SKILL=roll`，训练预算 10,000。

PPO、网络结构和课程参数见 `training-contract.json`，实际完成轮数见 `training-complete.json`，运行时源码哈希见 `source-manifest.json`。完整运行源码和中间检查点保留于服务器的 `roll823-10000-20261105` 目录。

## 最终仿真

823 g、7.4 V、两种子各 16 例，5 秒完整动作。成功需支撑累计转角至少 330°、获得头顶接触标记、最后 0.5 秒至少 90% 帧满足站稳判据，且无重置。

最终模型 30/32 成功，旧版 32/32。最终模型 32/32 无重置。逐例结果和旧版对照分别见 `results.json`、`baseline-results.json`。目前属于用户要求保留的最终候选，实机待测。

| 文件 | SHA-256 |
|---|---|
| policy.onnx | `f992714ea08e8c7a2157071160b077b87f5a5a7814b40b74957815b1db74edd4` |
| model_9999.pt | `ce862b3b6b05fe5ec8dce7845112fcbe4b360b3827f351f186609a2f278cab0b` |
