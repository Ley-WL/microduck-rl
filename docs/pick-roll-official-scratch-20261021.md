> **2026-10-22 拾取用户约束：** 用户再次明确“参考官方配方训练，不要随便改”。不采用前文建议的奖励宽度、容差或平滑课程消融，不自行改PPO/奖励/课程/周期/对称约束。只保留已授权HD1910及Observer限位适配；先审计公开任务、依赖与实际执行路径。已有20000轮完成且未通过，不将未启动的新任务称为正在训练；负载钩子问题先核验官方实现，不擅自改成非零奖励权重或新增训练逻辑。

# 翻滚 / 拾取：官方配方重新训练（2026-10-21）

用户要求两项按官方方法重新训练；从随机初始化开始，现役 Observer ONNX 仅用于对照，不导入 actor/critic/优化器，不自动部署或推送。

## 公开配方与适配

固定上游 `0bf989701f14d4008c866ffd6f09067870f99379`：[GroundPick](https://github.com/pollen-robotics/microduck_rl/blob/0bf989701f14d4008c866ffd6f09067870f99379/src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py)、[Roulade](https://github.com/pollen-robotics/microduck_rl/blob/0bf989701f14d4008c866ffd6f09067870f99379/src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py)。两任务配置和 symmetry.py 的AST与公开版本一致。[审计](pick-roll-official-source-audit.json)。拾取直接引用的24个MDP函数/类一致，翻滚28个中27个一致，reset_roulade_state仅增加环境原点z（平地z=0）；未声称完整依赖树逐字一致或复现作者某个未公开运行。

| 项目 | 拾取 pick | 翻滚 roll |
|---|---|---|
| 官方目标轮数 | 20000 | 10000 |
| 正式环境数 | 4096 | 4096 |
| 训练种子 | 211141 | 211151 |
| GPU | 0 | 1 |
| 任务方法 | 4秒周期，分段接近/保持/回升/站姿，模拟负载10–40g | 5秒回合，站姿和中途翻滚初始化，支撑条件下累计转角、头部接触标记与落地奖励 |
| 对称约束 | 官方关闭 | 官方开启 |

PPO、奖励、课程、事件、观测、动作缩放均沿用对应公开配置；适配当前replica几何、HD1910参数、Observer机械及动作目标限位，接触容量200/256，日志改为本地TensorBoard。拾取是嘴部接近地面并回升，不是带真实物体的抓取成功率。

## 启动与状态

入口：`scripts/server/train_pick_roll_official.py`；适配：`src/mjlab_microduck/tasks/pick_roll_observer_official.py`；6项配置测试：`tests/test_pick_roll_observer_official.py`。

独立服务器目录：`/CodeRepo/Code/leiwu30/hd1910-pick-official-20261021`、`/CodeRepo/Code/leiwu30/hd1910-roll-official-20261021`。复用已安装的Linux/CUDA Python `/CodeRepo/Code/leiwu30/1_code/microduck-training/microduck-rl/.venv/bin/python`，不安装或修改共享环境。

在独立源代码副本根目录放置对应现役模型为 `baseline.onnx`，使用已有环境执行：

```bash
OFFICIAL_SKILL=pick OFFICIAL_GPU=0 PYTHONPATH=src python scripts/server/train_pick_roll_official.py launch
# 另一独立目录执行
OFFICIAL_SKILL=roll OFFICIAL_GPU=1 PYTHONPATH=src python scripts/server/train_pick_roll_official.py launch
```

每项：配置测试→64环境5轮随机初始化试跑/导出→基线评估→正式从头训练。一次连续learn，不分段重建PPO；在1001/3001/5001/10001及最终轮数保存导出并评估。查看各目录 `evidence/progress.json`、`train.log`、`comparison.json`、`job.exit`；launch.lock阻止重复启动，失败保留现场，不自动重训。环境、基线副本、原始日志、NPZ与checkpoint不入库。

## 同条件筛查

`evaluate_pick_roll_official.py`：两种子211131/211139，各16世界；基线和候选初始qpos/qvel哈希必须匹配。正常工况固定7.4V、关闭推扰及部分随机化、actor观测无噪声。不是完整抗扰验收。

- 拾取：无负载12秒连续3周期；每周期接近窗口至少5帧嘴高[-5,25)mm且向下对齐>0.7，最后0.6秒至少90%帧倾角<20°、站高误差<20mm；全程无重置或倾角>60°。
- 翻滚：从站姿开始5秒，支撑累计转角≥330°、获得官方头顶接触标记、末0.5秒至少90%帧倾角<20°且站高误差<20mm，全程无重置。翻滚过程中大倾角属于任务，不能计为普通跌倒。

保存逐例成功布尔值、同初态哈希、首世界qpos及任务轨迹NPZ。正常筛查通过仍需回放动作风格、负载/抗扰和实物验证；不据此自动替换Observer。

## 启动记录

拾取launcher3207674、翻滚launcher3208126均已启动。两独立副本均6项配置测试通过（约37秒），正在进行64×5试跑。是否进入正式训练以progress和train.log为准。

翻滚首launcher3208126在试跑开始前因对称回调函数不能JSON序列化退出；仅修复配置记录（runner初始化前深拷贝原始配置），失败现场保留evidence-failed-contract-json。新launcher3234506重新通过6项测试，未修改奖励/对称/PPO。拾取launcher3207674无需此修复（官方关闭对称），保留启动时源版本；两者source-manifest分别记录。

拾取试跑5轮/120步、归一化count7680通过。现役拾取模型同条件筛查18/32成功，正在正式训练初始化。本筛查使用官方完整4秒周期，不能等同Observer当前动作时长下的实机结果。

## 正式训练确认（覆盖上文启动阶段）

两项均已通过6配置测试、64×5试跑与ONNX导出，完成基线32例评估：拾取18/32、翻滚30/32成功。最新确认正式4096环境拾取零基iteration77/20000、翻滚iteration30/10000，nan_state均0；actor/critic初始normalizer count0、优化器状态0，未加载checkpoint。日志预计拾取约7.5小时、翻滚约4.1小时（不含后续评估，非保证时间）。

本次新增文件为训练/评估入口、硬件适配、配置测试、来源审计和本文；README只更新现役两行的重训链接。源码和文档UTF-8无BOM/LF校验通过。没有提交推送或替换实机模型。

## 2026-10-22 最终结果与拾取审计

两项job.exit=0。拾取20000轮完成，各阶段0/32；翻滚1001/3001/5001/10000更新成功数32/28/30/28（分母32）。用户指定保留[翻滚1000轮](../models/roulade_hd1910_1000/README.md)，待实机验证，未部署。

官方拾取4秒周期：下降1.5秒、低位保持0.2秒、回升1.5秒、站姿休息0.8秒。嘴接近地面奖励权重3、目标0、高斯宽度100mm；朝下权重2，回升腿姿/颈姿各6、直立4；头撞地惩罚-2、阈值1N。平滑惩罚起始-0.8、250轮-1.5、500轮-2，随后增加重心随机化；PPO默认20000轮，512/256/128 ELU、在线归一化、std1、自适应LR1e-3。

最终完整周期最低嘴高的跨世界中位数52.6–55.7mm，与判定窗口一致，排除仅窗口错过低点；朝下对齐约0.98–0.995，返回站姿32/32。接近奖励未乘权重/相位部分exp(-(z/0.1)^2)：55mm仍约0.739，18mm约0.968。浅下探仍有较高收益，加上平滑/回站/碰撞代价，可能形成局部最优。这是机制假设，尚未进行奖励消融或逐关节限位诊断。原模型同硬件/限位能到约18mm，不能直接归因为机械不可达。

**负载更正：** 前文“保留10–40g模拟负载”仅指配置，不能等同实际施力。公开配置把apply_mouth_payload_force挂为weight=0的奖励；本次mjlab RewardManager.compute遇weight==0直接continue，项目NaN补丁仅调用原compute。质量虽抽样，该路径没有施力。本轮不能称已验证带载训练；这也不足以解释下探失败。

建议先记录逐关节限位裁剪、分项奖励及接触，单独验证负载钩子，再对接近奖励宽度/容差和平滑课程作小规模消融。届时需明确标为官方基础上的改进配方。本次仅审计，未启动新训练。

## 2026-10-22 深入核对：配置相同不等于完整复现

未修改奖励/课程/PPO、未重启训练。服务器原20k目录只读审计：任务配置与上游b02fdea4902c2b93b7d34a83a6f05da4ececb7d3 AST相同，mjlab1.3.0与官方固定版本相同。JointPositionAction先scale+offset再clip，未发现把绝对限位错误施加到raw action上的问题。

尚未复现作者发布权重对应的精确训练提交、种子、依赖锁及完整训练历史。对照18/32是Observer现用模型在本地筛查下的结果，不是作者官方成功率；32例为两个评估种子，同一个训练种子211141，不能证明从头训练稳定收敛。

官方机器人常量加载microduck/robot_groundcontact.xml，适配加载replica/robot_allcollisions.xml。进一步检查本地保留的两份XML：碰撞部件清单相同，总质量均0.73724318kg，mouth_tip位置相同，不能仅根据文件名声称多了碰撞体。尚未完成全部mesh和编译后碰撞矩阵等价证明。官方XL330 kp_fw200、电压6.5–8.2V/最低6V；当前HD1910 kp_fw5、电压7.4–8V/最低4V，以及Observer限位，是明确的动力学差异，P数值跨舵机不可直接比较。

保存的每个评估种子首世界最低嘴高时刻（不是全32世界统计）：

|模型|种子|最低嘴高mm|左/右膝°|左/右踝°|颈俯仰°|头俯仰°|
|---|---:|---:|---|---|---:|---:|
|现用|211131|16.32|47.1/-26.3|66.5/-67.8|-88.5|-22.2|
|现用|211139|16.74|47.8/-25.8|66.5/-67.5|-87.3|-20.7|
|新20k|211131|53.61|81.2/-78.8|67.7/-67.5|-85.6|21.0|
|新20k|211139|55.21|79.7/-77.8|66.8/-67.0|-87.0|24.3|

现用模型表明该硬件仿真/限位有更低的可达姿势。新模型膝/髋/头配合不同且踝接近边界，支持姿势策略不理想的解释，不证明限位是唯一原因，也不授权扩大机械限位。

TensorBoard最后一步19999：平均回报196.9866、平均episode length993.06；mouth proximity项0.71967、return legs2.21745、return neck2.14428、return upright1.52450、feet grounded2.98161、dof_pos_limits -0.11745。nan_state记录0。奖励高和长时间未倒不等于嘴到地；分项回报不能直接当作梯度贡献。payload weight0被RewardManager.compute跳过是执行路径问题，但官方配置/版本也有相同组合，不能直接断言只有本地遗漏或是浅下探根因。

后续应在官方配方不变前提下补齐原发布版本来源、完整依赖/执行器审计及接近阶段逐关节目标裁剪/接触诊断，再决定受控复现实验；不盲目重复20000轮或改奖励。

## 2026-10-22 冻结回放补充（含真正官方权重）

GPU4只推理，三策略×2评估种子×16世界，未优化权重。诊断脚本在原evaluator上增加每步raw action、HOME、q/qvel及reward分项采样。原始轨迹、日志、脚本位于服务器原pick目录evidence/diagnostic-frozen-20261022。汇总见[evidence](evidence/pick-frozen-20261022/analysis.json)。

官方发布[manifest](https://huggingface.co/pollen-robotics/microduck-policies/blob/main/manifest.json)给ground_pick周期4秒、结束phase0.7、实际2.8秒；没有该条目的训练commit/checkpoint。下载官方alpha_ground_pick.onnx并校验SHA ffbf5109982ff999b0ba53afe86b9ae731bbec679d67fb7f8ab4c52152c88872，确与现用2b69fa68不同。

|相同HD1910/Observer仿真|首次2.8秒嘴最低高度中位数（两seed）|原连续三周期判据|
|---|---|---|
|现用2b69fa68|18.09/18.32mm|20/32|
|20k新训20f4d43b|55.62/55.47mm|0/32|
|官方发布ffbf5109|35.80/35.50mm|0/32|

这不是官方原XL330场景评估，也不是完整2.8秒→行走交接测试，不能称官方模型本身失败。三者初态hash相同。新回放与历史轨迹不逐值一致：现用首世界qpos最大差约0.35（混合坐标单位）、新训约0.02–0.04，现用20/32对历史18/32不能算改善或独立统计显著证据；原因尚未隔离，保留所有不匹配标记。新模型浅下探现象重现。

下探窗口每周期1.0–1.8秒（含早期回升）：新训左右踝raw+HOME目标100%越界裁剪，头roll约95.9–96.1%裁剪，现用双踝也100%裁剪但头roll0%；因此目标饱和普遍存在，不能单独解释新训失败或授权扩限位。新训该窗口head_impact_penalty/self_collisions均0（奖励为0不证明所有接触均不存在）。新训膝目标与步后角度平均绝对差约9.6–12.1°、head_pitch约9.2°；包含运动相位/执行器延迟，不可当作静态误差或独立辨识参数。

结论：必须区分官方权重、现用供体权重、当前从头训练权重；此前现用对照不能当官方效果复现。下一有辨识力的实验是固定官方权重，分别测试官方原硬件、HD1910、再加入Observer限位的单因素对照，奖励/PPO不变；本轮尚未执行该三条件实验，无新训练或机器人操作。

## 2026-10-22 固定官方权重：执行器/限位/几何对照完成

本轮发现本地microduck/robot_groundcontact.xml不是固定上游原版：历史修改了左右hip_yaw、neck_pitch（-135°而上游-90°）、head_pitch（±105°而上游±90°）。不能再将这个本地路径当作官方基准。首诊断进程107054/107055已终止并保留日志，不计入结论。

重新下载官方b02fdea4902c2b93b7d34a83a6f05da4ececb7d3的XML（SHA99d23867e46e9d4748ff1398016b850f09fb6cf02a558c0272743b39ce3acf15），旁路文件robot_groundcontact_official_audit.xml，不覆盖训练输入。FrictionDRBamActuator源与固定上游逐字归一换行一致。共四条件×两seed×16世界=128例，官方权重固定ffbf5109；统一7.4V、关闭推扰/actor噪声/部分随机化，初态qpos/qvel hash逐seed一致，GPU4/5仅推理，job.exit=0 0。官方XML+XL330是当前库下重建的名义环境，不是发布权重的完整历史训练环境复现。

|条件|首次2.8s最低嘴高中位数，两seed|三周期本地25mm判据|
|---|---|---|
|固定官方XML+XL330|28.21/28.67mm|0/32|
|同XML只换HD1910执行器|39.64/40.77mm|0/32|
|HD1910再加Observer机械/目标限位|35.62/35.43mm|0/32|
|再换当前replica几何|35.60/35.50mm|0/32|

[汇总JSON](evidence/pick-hardware-pinned-20261022/comparison.json)。执行脚本原样保存于服务器scripts/server/compare_pick_hardware.py，本地.local副本；原NPZ/日志在evidence/pick-hardware-pinned-20261022，不入库。

本对照支持：完整执行器模型更换使这一固定策略下探变浅约11–12mm；Observer限位在本例反而降低嘴高约4–5mm，不能称限位为主要劣化因素；两几何版本加同限位后差<0.1mm，未见几何主因。执行器更换同时涉及识别模型、增益与Feetech状态实现，不等于证明扭矩不足或应调P。未改任何实机限位/增益。

验收纠错：官方基准也未达到自定25mm/连续三周期判据，因此0/32不能直接等同“没学会官方动作”。该指标仍可比较本地需求，但不得冒称官方成功率。另一方面，新训约55.5mm仍比同HD1910条件的官方权重35.5mm明显浅，不能仅以判据不同掩盖策略差距。

本轮没有优化权重、重新训练或访问机器人。剩余未闭合问题是官方发布权重的精确训练版本/运行合同，以及HD1910官方配方从头训练收敛到浅姿势的原因；不得以这次固定策略迁移试验直接证明训练失败唯一根因。
