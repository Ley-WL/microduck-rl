# 823 g 拾取 · 1,001 次更新

用户观看仿真后认可本候选，2026-11-05 要求提交保存。`policy.onnx` 为推理权重，`model_1000.pt` 为对应完整 PPO 检查点，编号从零开始。

## 训练方式

从随机初始化训练，官方 GroundPick 配方、4 秒相位周期、HD1910 执行器及 Observer 机械与动作目标限位，4096 个并行环境，seed 221149。各部件按 `walk_mass_823g.apply_mass_823g` 调整，编译总质量 0.823 kg。

本轮总预算为 10,000 次更新，本目录选取其中第 1,001 次更新的权重。完整网络、PPO、课程见 `training-contract.json`，其 iterations=10000 表示整轮预算。运行入口为 `scripts/server/train_pick_roll_official.py`，设置 OFFICIAL_SKILL=pick、OFFICIAL_SCREEN_UPDATES=10000、OFFICIAL_SEED=221149，并对 pick_spec 应用 823 g 配重。

## 仿真评估

同配重 7.4 V 名义环境，两种子各 16 例，每例连续三个拾取周期。32/32 达到下探及回站联合判据，32/32 无重置。首 2.8 秒最低嘴高中位数分别为 21.91、22.08 mm，旧版为 26.23、26.33 mm。回站窗口稳定帧比例均为 100%。

嘴尖最低时躯干下降中位数为 19.87、22.35 mm，旧版为 19.74、22.07 mm，身体下蹲幅度改善有限。联合判据包括嘴尖进入距地 25 mm 内、朝下对齐和回站稳定。实物拾取待测。

逐例结果见 `results.json`，旧版见 `baseline-results.json`，不同轮次对照见 `comparison.json`。最终 10,000 轮嘴高约 35 mm，此目录保留用户选中的早期权重。源码哈希见 `source-manifest.json`，完整运行源码保留于服务器 pick823-10000-20261105 目录。

| 文件 | SHA-256 |
|---|---|
| policy.onnx | `6e26c9deafe7812dd29dffa0247ef48ecdf796e3b9a710107f28371346ef644a` |
| model_1000.pt | `bbeaec8a029856a880ccbc6d57792815f0c6528dc84fc6d203c331fb93bf0de9` |
