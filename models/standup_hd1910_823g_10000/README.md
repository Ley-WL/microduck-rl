# 823 g 跌倒起身 · 微调 10,000 轮

2026-11-05 训练完成，按用户要求归档最终模型。推理文件为 `policy.onnx`，可续训的完整 PPO 检查点为 `model_9999.pt`。

## 训练方式

从现役 xgoduck_getup 对应检查点迁移 actor 和真实观测归一化统计，初始化新的 critic、优化器和课程计数。初始导出与旧 ONNX 的最大输出误差为 0。源权重哈希、网络、PPO 和迁移记录见 `training-contract.json`。

采用官方 Stand v1.5 配方，HD1910 执行器、Observer 机械与动作目标限位、0.823 kg 配重，4096 个并行环境，seed 261029，实际微调 10,000 次更新。训练合同中基础配方的 max_iterations=15000，实际运行预算由入口覆盖为 10000，完成计数见 `training-complete.json`。

运行使用 `train_standup_official.py`，将 `standup_observer_official.spec` 的返回值应用仓库的 `walk_mass_823g.apply_mass_823g`。本次运行源码哈希见 `source-manifest.json`，完整源码、迁移用 seed.pt 和中间检查点保留于服务器的 `standup823-10000-20261105` 目录。

动作合同为 HOME + raw action，scale=1，无输出 EMA。旧版对照采用其原来的旧值权重 0.45 EMA。部署时需要匹配对应观测、历史动作和执行合同。

## 最终仿真

坐姿、站姿、俯卧、仰卧，两个种子，每组 16 个环境，共 128 例。

新旧均 128/128 曾达到起身交接条件，最终模型无重置或无效状态。末段颈部 HOME 偏差均值由旧版 1.15° 增至 8.92°，头部回正退步。逐例偏差与交接结果见 `results.json` 和 `baseline-results.json`。目前属于用户要求保留的最终候选，实机待测。

| 文件 | SHA-256 |
|---|---|
| policy.onnx | `e14d8f4d6d80b02efefbbb23b69b27e244de35ef92dd4ed65ae8305f9fa6dd3d` |
| model_9999.pt | `4c777e7c9a8048eb08665b8adc3f024d393a5f015b103f4a18b8545244d84943` |
