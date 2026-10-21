# 官方配方 + observer 限位 + HD1910，从头训练

用户于 2026-10-20 明确要求重新训练。本轮独立目录：
`/CodeRepo/Code/leiwu30/hd1910-walk-observer-20261020`。

## 固定配方

以官方 develop `b02fdea4902c2b93b7d34a83a6f05da4ececb7d3` 对照核验过的
velocity 配方为基线（任务配置 Git blob `45a1f6d5b7f5d81aa58dab777ed34f458a2f847c`）。
保留官方奖励、指令采样、课程、观测噪声和随机化、PPO adaptive LR 1e-3、
24 步、5 epoch、4 mini-batch、std 1、可更新的归一化。完全从头训练，
不载入旧 ONNX、critic 或 optimizer，不加教师保持、不增加髋摆动/限位奖励。
预算 4096 环境 × 5000 轮，每 250 轮保存；先做 64 环境 × 5 轮冒烟。

## 硬件适配

复用 HD1910 `1910_m6.json`、P5、7.4–8.0V、电压下垂和 3–6 仿真步延迟；
复用已有 replica 几何/惯性，不声称本次完成硬件辨识。
保留官方基础 walk 接触选择，不引入 backlash 或额外身体接触。

关节范围取 observer 后端 `backend-rust/src/telemetry.rs::limit`，
不是前端陈旧 `model.json`，与已有 `getup_mechanical_contract.LIMITS` 一致：

|关节|角度|
|---|---|
|左髋偏航|−25°～30°|
|右髋偏航|−30°～25°|
|髋滚转|±22°|
|髋俯仰、膝|±90°|
|脚踝|±70°|
|颈俯仰|−90°～70°|
|头俯仰|±90°|
|头偏航|±170°|
|头滚转|±20°|

这既用于 MJCF 实际关节范围，也用于 HOME + scale 后的目标裁剪；
使用 mjlab 原生 JointPositionActionCfg.clip，保留原始动作历史、scale 1、无 EMA。
目标裁剪属于 observer 硬件执行约束，明确区别于官方宽目标范围。
本轮不增加编码器取整、指令平滑等运行近似；这些由共同运行仿真评估。
未连接实机重新验证 EEPROM/标定，所以这里“可达”指后端已配置机械范围，
不等同于新的实测可达性证明。部署前仍须校验目标机器编码器与 EEPROM。

## 最终模型与验证

5000轮训练已完成（checkpoint索引4999）。最终部署模型与运行metadata已保存于 `models/walk_observer_hd1910_5000/`，SHA256为 `e3b90ac0e5359f91e14306b7e2143e0d5a1f4cd49cb256bca3fd71835250d895`，与Observer采用的模型完全一致。用户于2026-10-20及21反馈效果很好。10000轮候选没有全面超过本版本，未替换现役。模型包为部署ONNX，不含恢复Adam/critic所需checkpoint。

## 复现命令

使用仓库pyproject.toml/uv.lock对应的Linux CUDA环境，安装方法见主README。新checkout中运行：

```sh
PYTHONPATH=src python tests/test_walk_observer_contract.py
python scripts/server/train_walk_observer.py launch
```

启动器默认GPU4，要求空闲显存大于30000MiB；单卡机器可修改launch资源选择，保持训练超参数。先64环境5轮冒烟，再4096环境5000轮。evidence/保存输出，不入库；launch.json防重复，source-manifest冻结源码。不自动部署。

提交版训练后自动评估为最终导出单模型30输入，去掉对服务器相邻旧实验目录的依赖；只改变评估包装，未改变训练配方。历史首次评估为三模型20输入，之后另外补做30输入速度对照，结果见模型包simulation-comparison.md。

独立评估已提交模型：

```sh
PYTHONPATH=src MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/compare_walk_inputs.py
```

可用WALK_POLICY_PATH、WALK_OUTPUT_DIR指定同合同模型和输出。默认HOME来自模型包；不同HOME需用MICRODUCK_WALK_SURFACE_INPUTS指定含walk.metadata.json的目录。本模型标签默认observer5000；旧donor EMA需显式initial标签并指定模型。

两项合同测试验证奖励/课程/指令/观测/事件/终止未变、14关节范围、HD1910/P5、裁剪顺序及raw history。每输入16环境10秒，2–8秒输入，4–8秒统计；倒地后屏蔽数据。共同运行近似包含backlash、身体接触、7.4V、14ms延迟、编码取整、前一帧反馈速度、指令平滑及理想gyro；这是评估工况，不是训练设置。

历史30输入评估480/480未倒，但原地小幅左转弱响应及后退偏航仍待优化。用户反馈不等于完整量化验收；metadata.hardwareValidated=false保留。部署合同61→14、50Hz、scale1、无EMA、P5/D0；目标机器编码器/EEPROM可达性仍由控制器校验。
