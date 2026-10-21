# Flamingo Cycle：HD1910 微调与仿真

用户要求基于 RemiFabre/microduck-flamingo-cycle，在当前仿真参数下微调并先看仿真；本轮不部署、不连接机器人、不改行走模型。

## 来源与训练方法

模型页：https://huggingface.co/RemiFabre/microduck-flamingo-cycle
上游源码：pollen-robotics/microduck_rl @ 0bf989701f14d4008c866ffd6f09067870f99379，flamingo分支，Apache-2.0。
Hub固定revision：a69e4f6103a70e355983eb5d4e919da76778a40f。
ONNX SHA256：df77929c39d7695092bdaf810c2075e20a9ba91abd8192b4073d3de593d56904。

原模型为XL330仿真，61维观测→14动作、50Hz、scale1、无EMA，normalizer内嵌。twist=[flag,side,0]：flag0双脚站立、1单脚；side+1抬左脚、-1抬右脚。奖励内部alpha按1.5秒斜坡变化，观测使用原始flag/side。阶段1学习单脚保持，阶段2加入双侧HOME↔单脚循环与混合初态，Hard版本后期推扰至0.25m/s；不是硬编码动作轨迹。

## 本轮适配

沿用当前replica几何/惯性和HD1910 BAM 1910_m6/P5、训练7.4–8.0V及压降/3–6步延迟；关节物理范围及目标裁剪使用Observer LIMITS。保留原FlamingoCycleHard奖励结构、命令与成熟课程，不附加行走奖励。原型及循环配置与MDP helper按固定commit导入，独立命名，不覆盖现有mdp。

从ONNX恢复actor MLP、观测均值/std；count合成为1000000并冻结actor normalizer。原critic/Adam不可从ONNX恢复，所以新建critic/Adam，先10轮只更新critic并逐值确认actor不变。随后低LR1e-5、探索std0.08，64环境5轮冒烟，再1024环境100轮短微调；其余PPO结构保留。不是原run完整resume。

独立服务器目录：/CodeRepo/Code/leiwu30/hd1910-flamingo-cycle-20261021；使用既有训练venv。启动器先检查空闲GPU并写launch.json防重复。首次GPU5资源门未通过，未启动；改用GPU2。状态以evidence/progress.json及job.exit为准。

## 仿真判据

原模型/微调模型各6场景×16环境：左右抬脚，每侧无推扰、向后0.1m/s、横向0.1m/s。16秒序列：0–2双脚，2–11单脚，11–16落脚。统计4–10秒单脚保持占比≥80%、14–16秒双脚接地且倾角<0.25rad占比≥80%；终止或倾角>55°锁定失败，不能用重置后片段冒充成功。保存同种子初态qpos/qvel哈希、目标帧差及左右名义场景视频。

评估采用7.4V、固定14ms执行器延迟与当前限位；不是实机测试，执行器参数未重新辨识。模型卡原推扰能力不能直接当HD1910结果。完成后填写基线/候选指标、视频判断及是否接受。

## 运行记录

硬件/镜像合同2项通过。启动器947409，GPU2。前两次基线在辅助函数导入及JointPositionAction字段名处失败，未正式训练，日志分别保留evidence-failed-import和evidence-failed-api。已补齐上游私有helper导入，并用当前mjlab的_processed_actions及write_root_link_velocity_to_sim；推扰按机身偏航旋转到世界系，失效后样本不计入目标变化/抬脚高度统计。

评估时延纠错：上游默认步长5ms，最初7步实际为35ms，该批数字无效、仅留evidence-invalid-35ms，不用于比较。已只停止本任务进程组，重跑launcher979848；评估明确2ms×10=50Hz、7步=14ms，头部指令归零。训练仍保留官方5ms×4=50Hz及当前HD1910 3–6步（15–30ms）随机执行器延迟。

## 最终结果：短微调退步，不采用

launcher979848 已完成，job.exit=0。1024环境100轮PPO（另10轮critic预热），候选SHA256 `4df12bb6bbe2694bc64ec2a05fe3e8f52ff5329772664e29f88ace30eedac5ac`。原模型与候选均在相同HD1910参数下评估；6组初始qpos/qvel哈希逐组相同。每组16例，以下是单脚保持成功数：

| 场景 | 原模型 | 微调100轮 |
| --- | ---: | ---: |
| 抬左脚，无推扰 | 13/16 | 11/16 |
| 抬左脚，向后推扰 | 15/16 | 9/16 |
| 抬左脚，横向推扰 | 14/16 | 11/16 |
| 抬右脚，无推扰 | 16/16 | 10/16 |
| 抬右脚，向后推扰 | 16/16 | 10/16 |
| 抬右脚，横向推扰 | 16/16 | 10/16 |
| **合计** | **90/96（93.8%）** | **61/96（63.5%）** |

两版均96/96未倒、96/96按时落脚；微调丢29个原成功案例、无新增成功案例。无推扰保持阶段平均抬脚高度：左7.53→6.16cm，右10.47→7.04cm。对应全周期目标帧差P95左2.86→2.70°、右2.57→1.99°；目标变化变小伴随保持退步，不能称为控制改善。

代表环境视频关键帧已检查：两版均有抬脚和落脚动作；视频只录每组第一个环境，该环境成功不代表16环境全部成功。并排视频左列原模型、右列微调，上排抬左脚、下排抬右脚，16秒完整周期。批量结果以上表为准。

视频对应的第一个环境也有明显抬脚延后：从2秒发出抬脚指令算起，首次连续0.5秒满足单脚判据，左脚原版0.86秒/微调3.14秒，右脚原版0.78秒/微调2.48秒；4秒截图微调版仍未形成有效单脚保持，8秒已抬脚。此项仅是代表环境时序，不是16例平均。

本次观察到动作幅度收缩和单脚保持退化；尚未通过对照实验确定具体原因。可能相关因素包括新critic价值估计、低探索、硬件改变后直接启用成熟Hard课程；不能把这些可能性当作已证明原因。后续建议先用温和推扰适应硬件、加入原actor约束以减少遗忘，再逐步恢复Hard课程，并在未用于调参的种子上验证。本轮不继续盲加轮次、不部署机器人。

本地证据根目录：`artifacts/flamingo-cycle-hd1910-20261021/`（实验产物被git忽略）：

- `comparison-h264.mp4`：原版/微调版左右抬脚对照视频。
- `summary.json`：逐组结果、配对丢失/新增数及候选SHA。
- `final/evidence/{baseline,candidate}/`：results.json、contract.json、6组全帧NPZ及名义场景视频。
- `final/evidence/refine100/{policy.onnx,final.pt,complete.json}`：候选推理模型与训练checkpoint。
- `trial-complete.tar.gz`：源码片段、输入与完整有效运行证据；SHA256 `32b6991f86989dbd11373532546ebbcc627d698593fdb681f3b574f0ea30b70a`，下载后已校验。

验证：7个Python文件AST通过、2项硬件/镜像合同测试通过、候选ONNX与恢复actor随机观测最大误差4.77e-6、normalizer训练前后逐值不变。已有行走模型和机器人均未修改。本轮只是当前仿真近似下的短程微调，不是实物效果证明，也未建模完整实机编码器/回差链。
