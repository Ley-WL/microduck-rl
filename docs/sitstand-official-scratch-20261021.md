# 坐站：官方配方从头训练

用户于2026-10-21选择从头训练，当前Observer坐站ONNX仅作对照，不作为初始化权重。现役模型不替换。

## 最终状态（2026-10-21）

用户要求提前停止，训练进程已退出；最后日志零基iteration4299，最新保存model_4250.pt。发布已评估的model_3000.pt（实际3001更新）导出模型，未部署。

[模型与测试结论](../models/sitstand_hd1910_3000/README.md)：1001更新56/64成功，3001更新64/64成功且64/64未倒，原模型57/64成功、58/64未倒。4250检查点未评估，不把其效果等同3000模型。原计划15000轮未完成，以下启动/计划记录为历史。

## 配方与差异

- 固定上游：`0bf989701f14d4008c866ffd6f09067870f99379`，`microduck_sitstand_env_cfg.py`本地与公开源AST完全一致。[官方配置](https://github.com/pollen-robotics/microduck_rl/blob/0bf989701f14d4008c866ffd6f09067870f99379/src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py)。此校验不代表完整依赖源码逐字相同。
- 公开奖励、课程、12秒episode、坐站命令、2秒奖励目标渐变、PPO均保持。MLP512/256/128、ELU、在线归一化、std1、学习率1e-3 adaptive、24步/环境、5epochs/4minibatches，默认15000轮。
- 适配当前replica几何、HD1910 P5/电压/延迟模型、Observer机械和动作目标限位；接触容量200/256。没有新增磕头惩罚，官方允许过渡中头部辅助支撑，不保证解决实物磕头。
- TensorBoard本地记录、seed211121、GPU1为本轮执行选择。64环境5轮冒烟和4096环境正式训练分别随机初始化；无ONNX导入、无借用critic、无合成normalizer count、无自定义critic预热。

## 入口和对照

训练：`scripts/server/train_sitstand_official.py`；适配：`src/mjlab_microduck/tasks/sitstand_observer_official.py`；回归：`tests/test_sitstand_observer_official.py`。

服务器独立目录：`/CodeRepo/Code/leiwu30/hd1910-sitstand-official-20261021`。复用Python：`/CodeRepo/Code/leiwu30/1_code/microduck-training/microduck-rl/.venv/bin/python`。

当前基线SHA：`1c94934c6421ec72ed48079cea74966868a7a59f7d732e0207d0f7bdaeed3c50`。公开KissQy清单指向原`model_10000.pt`，公开导出包只有ONNX/参数/清单，不含训练checkpoint；用户已选择不依赖该checkpoint。

顺序：配置回归→64×5冒烟→基线评估→4096×15000正式训练。在1001/3001/5001/10001/15000次更新导出并与基线配对；训练使用一次连续learn，保持原save_interval250。

评估：`scripts/server/evaluate_sitstand_official.py`，站起始循环、坐起始循环、站保持、坐保持，各16环境，24秒四段；每段末2秒至少95%帧满足倾角<20°、高度距目标<12mm、腿姿平均误差<0.25rad，全程无重置/倾倒>60°。候选与基线初始qpos/qvel哈希必须一致。此为单seed正常工况64例筛选，不是抗扰或实物验收；轨迹保留NPZ供复核。

状态文件：`evidence/progress.json`、`launch.json`、`smoke.log`、`baseline.log`、`train.log`、`comparison.json`、`job.exit`。launch.lock拒绝重复启动，失败不自动换配方或重训。当前启动状态以运行记录为准，不凭脚本存在声称已训练。

## 源码审计补充

任务直接引用29个MDP函数/类，26个AST与固定上游相同；3处本地兼容差异：SitStandCommandCfg新增alternate=false默认，posture_height_l1增加error_scale=1默认，reset高度加环境原点z。本任务未启用alternate/缩放，平地原点z=0；没有借此修改奖励或姿态目标。完整依赖链仍非逐字上游镜像。

配置回归3项通过；2026-10-21已启动launcher1616835/GPU1，当前先执行smoke与基线，正式训练是否开始以progress/train.log为准。

## 基线筛选已完成

现役模型在本轮同硬件/限位的正常筛选工况：站起始循环15/16、坐起始循环13/16、站保持16/16、坐保持13/16，合计57/64；未倒58/64。只能与本轮相同脚本/初态的候选比较，不能和旧CPU/GPU验收混算。smoke已完成5轮/120环境步，normalizer count7680，之后已进入train阶段初始化。

## 正式训练已确认

launcher1616835、训练子进程1648560，GPU1；日志确认零基iteration27/15000，nan_state=0，约1.48秒/轮，日志ETA约6小时18分（不含后续评估，非保证时间）。正式contract为4096环境、15000轮；actor/critic归一化计数0、优化器状态空，从头开始。progress仅保存模型时更新，实时轮数看train.log。

## 复现入口

在独立副本中将当前 Observer 原坐站模型保存为仓库根目录 `baseline.onnx`，核验SHA为上文基线值，然后使用已有Linux/CUDA环境执行：

```bash
PYTHONPATH=src python -m pytest tests/test_sitstand_observer_official.py -q
SITSTAND_GPU=1 PYTHONPATH=src python scripts/server/train_sitstand_official.py launch
```

入口仍采用官方15000轮默认目标；本次提前停止属于用户选择，不改写官方配方。不要在已运行目录重复启动。训练环境、基线副本、原始日志与完整checkpoint不入Git。
