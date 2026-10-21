# Flamingo：参考公开课程重新微调

用户于2026-10-21确认从原模型重新开始，参考作者公开配置。上一轮短微调4df12bb6保持61/96、低于原版90/96，已拒绝采用。本轮输入仍为原模型df77929c，不从失败候选继续。

## 配方及明确差异

来源为pollen-robotics/microduck_rl固定commit `0bf989701f14d4008c866ffd6f09067870f99379` 的FlamingoCycleHard；复用已下载并核验的源码与HF模型revision `a69e4f6103a70e355983eb5d4e919da76778a40f`。

- 完整保留公开奖励权重、PPO算法参数和课程时间点，不把课程压缩到终点。
- 学习率初始1e-3、自适应，探索std初始1.0；网络512/256/128 ELU，每环境24步，5 epochs、4 minibatches。
- 推扰：0轮为0，500轮0.08m/s，1000轮0.15m/s，1400轮0.25m/s。
- 动作变化惩罚初始-0.1，600轮变为-0.5；力矩变化惩罚800轮开启；初始单脚姿态占比0.6→0.4→0.3，按原600/1200轮调整。其余重心随机化课程也保留原安排。
- 当前replica几何/惯量、HD1910 P5/1910_m6、电压/延迟随机化与Observer机械/动作限位保持；训练5ms×4=50Hz。仿真接触缓冲200、匹配容量256。
- 恢复ONNX actor及观测mean/std；ONNX不含critic、Adam及归一化count，故新建critic/Adam、count合成1000000。先10轮仅预热critic且断言actor逐值不变，之后允许actor归一化更新。该预热不消耗正式课程轮数。
- 正式4096环境、最多3000轮；64环境5轮冒烟，再从完整checkpoint续到7轮，验证续训逻辑。正式段从原ONNX独立开始，不使用冒烟权重。
- 因只有发布ONNX，本轮是已训练cycle策略的HD1910迁移，不是原作者从stage1到stage2的完整训练复现。4096为本轮并行环境选择，不声称已确认原发布run的实际环境数。

## 分段与评估

正式训练在100/500/1000/1500/2000/3000轮保存checkpoint、导出ONNX并各跑6场景×16环境。每段续训恢复actor、critic、Adam、normalizer及common_step_counter；保存轮数采用已完成更新数，避免runner零基索引造成重复。分段间重建仿真环境，因此不是逐帧无中断轨迹复现。

复用上一轮已完成的原模型基线，仍用同一评估入口、14ms延迟/7.4V/50Hz、初态seed211031；每个候选与基线初始qpos/qvel哈希必须相同。训练seed211022。固定筛选集已被查看，不当作独立验收集；若有改善，之后仍需未用于调参的新种子验证。

每组记录单脚保持、未倒地、成功落脚、目标帧差、抬脚高度及视频。中途指标只供筛选；即使高于原版也不自动部署。程序异常/NaN/已识别接触溢出时结束并写job.exit=1，不重复启动。全部完成也仅置needsReview，不覆盖机器人模型。

## 文件与运行

- 配方：`src/mjlab_microduck/tasks/flamingo_hd1910_curriculum.py`
- 训练：`scripts/server/train_flamingo_curriculum.py`
- 阶段调度：`scripts/server/run_flamingo_curriculum.py`
- 配方一致性测试：`tests/test_flamingo_curriculum.py`
- 服务器独立目录：`/CodeRepo/Code/leiwu30/hd1910-flamingo-curriculum-20261021`
- 复用环境：`/CodeRepo/Code/leiwu30/1_code/microduck-training/microduck-rl/.venv/bin/python`，未安装新依赖。
- 状态：`evidence/launch.json`、`progress.json`、`comparison.json`、`job.exit`；详细训练/评估日志与模型按阶段保存。

在上述服务器目录检查配置（不启动训练）：

```bash
PYTHONPATH=src /CodeRepo/Code/leiwu30/1_code/microduck-training/microduck-rl/.venv/bin/python -m unittest discover -s tests -p 'test_flamingo*.py'
```

4项测试已通过，包括课程/奖励/PPO逐项对比及硬件/镜像合同；4个新增Python文件AST、UTF-8无BOM/LF检查通过。GPU1资源门要求空闲显存>32GB、利用率<15%。日志会保留迭代耗时，实际显存与启动结果在运行确认后补充；没有新增推理性能宣称。源码未提交推送，环境与实验产物不入库。

运行确认：当前launcher1401494，GPU1；64环境5轮冒烟及恢复checkpoint续至7轮均通过，正式4096环境train100已启动。首次launcher1386287仅在预热后环境reset时触发inference tensor上下文错误，未正式训练，证据归档`evidence-failed-reset`；修正为在inference_mode内重置后重新通过冒烟。复用基线policy SHA已再次核验为df77929c。尚无本轮正式仿真改善结论。

首次正式进度确认：日志到零基iteration24/100（已执行25次正式更新），输入actor恢复误差5.25e-6；阶段合同确认4096环境、初始LR0.001/adaptive。一次采样GPU1总占用22141MiB、空闲26379MiB、利用率69%（包含共享GPU上其他占用，不等于本进程峰值）；当时迭代耗时1.49秒。没有NaN终止记录。

## 中途检查：约2038/3000轮，明显退化

用户询问进度时，launcher1401494仍运行，阶段train3000，最近日志零基iteration2037/3000。100、500、1000、1500、2000轮的96例保持成功均为0；未倒地依次96、96、96、50、80，落脚成功依次70、96、96、50、80。原版基线保持90/96、未倒/落脚均96/96。因此本轮截至2000轮没有改善，且比前一短微调更差，不能采用或部署。

2000轮同初态评估的无推扰平均摆动脚高度左约0.2cm、右约3.4cm（右侧也未满足单脚接触与持续时间判据）；表明动作未形成合格单脚保持，不能用未倒地冒充动作成功。候选ONNX/恢复actor误差3.81e-6，训练计数48000=2000×24正确；暂未发现该评估中的导出数值不一致。训练normalizer count为197608000，学习率约1.139e-4。具体退化原因未完成消融验证。评估contract内lr/std字段是评估器初始化值，不是该候选的训练参数，训练参数以train2000记录为准。

本地阶段汇总：`artifacts/flamingo-published-curriculum-20261021/comparison-through-2000.json`。计划内3000轮与末次评估仍继续，没有另启训练，也未改机器人。
