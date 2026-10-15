> **2026-10-16 v40结束/未通过，v41全运动均值评分冻结审计活跃：** v40两版GPU264/264无丢例，但低速起步77→61/96、脚步2.633→1.649cm，±.2速度偏差.01455→.04052、脚步10.48→8.35cm；CPU36/36且36初态/首velocity/指令配对，低速10→6/12、四纯±.05不启动，±.2实际约±.16、脚步8.33cm，拒绝加轮。critic训练估计接近目标但步态仍退化；job0/source-check/归档d5444b01下载核验。独立walk-motion-reward-v41-20261016 launcher3389446/worker3393080已确认活跃，冻结v28/v40，新237151/237157，±.05/.2/.4×3yaw×4，同轨迹旧/新所有运动指令EMA速度误差评分（stop/低速原评分精确保持）。真实mjlab合成评分/配置/legacy记录/reset检查通过：旧偏好慢稳、新偏好准确振荡；尚无真实轨迹排序结论或新奖励训练，勿提前开训。无PD/实机/部署，目标未完成；见docs/validation/20261016-walk-warm-critic与docs/walk-guided-training.md。

> **2026-10-16 v39完成通过价值门/v40动作微调活跃：** v39两seed各48/48完整60s，实际平均回报18.704/18.586、预测17.930/17.893，RMSE2.173/2.144、EV .600/.633；预设整体偏差20%/RMSE30%/EV>.3及四运动停止阶段门10/10通过，仅critic准备门、非步态验收。actor全state与v28逐值相同/导出b0809f25，float64回报重算误差<4.25e-5，job0/source-check/终止PID/归档7261ebc3下载通过。独立walk-warm-critic-v40-20261016 launcher3322754/worker3327987已确认活跃，v28actor+仅v39critic载入（SHA5286e66f），空Adam/iteration0/LR1e-5/anchor5/原200轮anneal计划，其它奖励/PD不变；658源hash通过，64×5/40Adam/finite/normalizer/导出通过，1024×200在途，之后236151/236157宽速度确定性筛选。勿重复启动，无实机部署，目标未完成。详见docs/walk-guided-training.md及docs/validation/20261016-walk-critic-warmup。

> **2026-10-16 v38结束/critic低估证实，v39仅价值预训练活跃：** v38两个seed各48/48完整60s、actor/critic逐值未变，首30s实际折扣回报18.161/18.143，预测8.936/9.145，RMSE9.742/9.489、EV .114/.156；CPU float64重算误差<4.53e-5，job0/source-check/归档c49ec57d核验。诊断延长episode64s有边界，非唯一梯度根因证明。独立walk-critic-warmup-v39-20261016 launcher3262330/worker3267654已确认活跃，v28actor全冻结/新critic LR3e-4/seed68131/原32s环境；64×5/40Adam冒烟通过，actor state逐值未变、actor Adam state0、导出SHA仍b0809f25，1024×200进行中。结束只做冻结价值校验，不自动actor训练。无PD/实机/部署，目标未完成，勿重复启动；详见docs/walk-guided-training.md及docs/validation/20261016-walk-critic。

> **2026-10-16 v37结束/v38冻结价值校验活跃：** v37原/候选各48/48，50偏航退化阶段0/50候选得分更高（7.6582→6.8260），主要tracking_bias惩罚扩大，不支持偏航奖励倒置或盲加yaw权重；job0/source-check/归档556aa180下载核验。独立walk-critic-audit-v38-20261016 launcher3232329已确认活跃，v36渐减actor+critic冻结、原观测噪声/训练分布采样，新235151/235157，±.05/±.2×3yaw×4、60s轨迹前30s价值对实际折扣奖励；episode延至64s避免32s超时截断，仅诊断。无更新/PD/实机/部署，先测critic质量再定预热，不重复启动。目标未完成，详见docs/walk-guided-training.md及docs/validation/20261016-walk-yaw-reward。

> **2026-10-16 v36结束/未合格，v37冻结奖励诊断活跃：** 两组1024×200/1600Adam/finite/normalizer/导出/std审计通过；固定组GPU低速起步80→72/96、yaw .047→.278，渐减79→80/96、yaw .0468→.1333，步幅未改善。CPU均36/36、72初态/首velocity/指令精确配对，渐减+.05实际仅.008–.009，±.2 yaw误差.0565→.2089且停止退化；两组不加轮。GPU同seed前期权重有数值差异，勿作逐位配对或noise唯一因果结论。v36 job0/source-check/终止PID、227007470字节归档c5f97757下载核验。独立walk-yaw-reward-v37-20261016 launcher3183640/worker3183645已确认活跃，冻结v28与v36渐减actor，±.2×yaw0±.6×4×新seed234151/234157，同mean_tracking奖励逐阶段审计，查偏航轨迹评分偏好；无优化器更新/实机/PD/部署。勿重复启动，目标未完成。详见docs/validation/20261016-walk-noise及docs/walk-guided-training.md。

> **2026-10-16 v36完整循环/探索渐减对照已启动：** 独立walk-noise-transfer-v36-20261016，launcher3064186；v28step500 actor/新critic/空Adam/冻结normalizer/相同seed与低速均值奖励，两组分别固定std.05与前65轮.05、第65–129轮线性降至.005、后70轮保持；两组各64×5冒烟后1024×200，96s名义轨迹覆盖32s回合，最后低噪声阶段33.6s。分布参数冻结，仅新rollout开始改变；真实Gaussian两种模式各200轮×24采样及update重算log概率逐值一致通过。当前fixed-smoke，效果尚未评估，不重复启动。新seed233151/233157宽速度确定性筛选；不以随机采样作为运行端修复，不改PD/实机/部署。详见docs/walk-guided-training.md，目标未完成。

> **2026-10-16 v34失败/v35采样依赖诊断完成，无活跃作业：** v34各264/264但低速起步80→77/96、yaw偏差增大，CPU36/36但低速8/12、后退停滞/前进约.01，未加轮。job.exit0/source-check/归档c16c4492核验。v35同v34权重，确定性/训练分布采样各48/48，专用RNG不消耗物理随机数；纯直行起步13→25/32，全低速74→87/96，但stop .00604→.01013m/s、yaw残速.03024→.08934，不能部署随机输出。归档c6bcb807/job0/sourcecheck完成。另代码证实50轮×24×.02=24s未覆盖30s循环、日志mean reward只统计done回合，不用该短程均值推断全部环境。下一考虑探索噪声渐减与完整循环采样，仍确定性输出验收/大步目标，不再只改奖励；无PD/实机改动。详见docs/walk-guided-training.md。

> **2026-10-16 v33验证通过/v34均值跟踪微调运行：** v33同轨迹12个停滞→准确阶段，旧评分仅1/12偏好行走（8.788对7.616），新评分12/12（8.920对11.306）；stop差<1e-6为旧记录器缺dt乘除的浮点舍入，非奖励函数差异，未来记录顺序已修。job.exit0/source-check/归档e46c2289核验，无学习。独立walk-mean-tracking-v34-20261016 launcher2932557活跃，645源hash通过/355控制源input保持；64×5/40Adam/finite/normalizer/61→14误差2.62e-6通过，1024×50在途，同v28actor/seed68131/新critic/空Adam/LR1e-5/anchor5，仅低速EMA评分改变。GPU231151/231157宽速度筛选，原v30Control复用不重训；实际步幅/起步/停稳未验收，勿重复启动。无PD/实机/部署，目标未完成，见docs/walk-guided-training.md。

> **2026-10-16 v32奖励倒置已证实/v33冻结新评分验证中：** v32原/教师各48/48，job.exit0/source-check/归档960a23cc核验。14个学生停滞而教师准确行走阶段，12个原奖励反而偏好停滞，均总加权率8.666对7.744；主要瞬时linear4.093对1.901/angular.843对.215，平均偏差惩罚虽改善但不足抵消。不是已学会新步态或唯一训练根因。新增仅低速.03<|v|<.1用.7s EMA三轴速度误差评分，非低速/停止/pureyaw原函数保持；合成振荡/稳态偏差/部分reset及配置对照通过。独立walk-mean-reward-v33-20261016 launcher2889589/worker2893364活跃，无优化器更新；新seed230151/230157，逐轨迹同时累计新旧加权得分，先验证激励再决定训练。PD/实机/部署不变，目标未完成。见docs/walk-guided-training.md。

> **2026-10-16 v31结束/未通过，无活跃训练：** 1024×50/Adam400/finite/normalizer/导出通过，job.exit0/source-check及归档d150b957下载核验。GPU原/候选264/264，无丢例但低速起步76→64/96，vx/yaw偏差.01770/.04326→.02308/.08844，步幅2.59→2.05cm；CPU36/36存活但低速8/12，四纯直行均不迈步，±.2转向误差.05649→.20248，未加轮/部署。仅延长奖励窗口没有形成慢速大步，下一直接审计冻结低速不走/实际走路的奖励分项与学习激励，避免继续只调步幅相关次要奖励。证据docs/validation/20261016-walk-air-window；PD/实机不变，目标未完成。

> **2026-10-16 v31摆腿窗口对照运行中：** walk-air-window-v31-20261016/launcher2776745/worker2785459已核验活跃；638源hash保持。仅低速.03<|v|<.1将air_time上限按.06/(2v)延长，.05为.6s，高速/零速/纯yaw逐值保持，权重3与其它奖励不变（大步仍1）。实际mjlab奖励边界及配置对照通过，64×5/40Adam/finite/normalizer/61→14导出误差4.77e-6通过，1024×50在途。仍v28actor/新critic/空Adam/LR1e-5/anchor5，同seed68131；v30control复用355份src/input及训练器仅allowlist差异核验，不重训控制组。GPU新seed228151/228157宽速度筛选；无实机/PD/部署，目标未完成，勿重复启动。详见docs/walk-guided-training.md。

> **2026-10-16 v30结束/未通过，无活跃训练：** 两组1024×50/Adam400/finite/normalizer/导出均通过，job.exit0/source-check/归档8a24e920下载核验。Control GPU丢227157/world78（+.05/0）；Stride无丢例，但低速起步79→76/96且速度/yaw误差增大，步幅仅2.53→2.60cm。CPU三版各36/36，低速起步基线10/12→两组8/12，纯直行基本不启动，均未续训。发现原air_time权重3仅.125–.3s给分；理想低速6cm约需.6s摆腿，现窗口单位时间收益减半，属于奖励冲突证据而非唯一根因。下一考虑按速度调整低速摆腿奖励窗口，保留高速度/原地转向/停止行为，再短程对照；不只加权或加轮。完整指标/同机CPU初态72对核验见docs/validation/20261016-walk-focused-stride。无实机/PD/部署，目标未完成。

> **2026-10-16 v30低速大步对照启动：** 独立walk-focused-stride-v30-20261016/launcher2676320活跃，634源码输入hash核验；v28step500 actor为未合格起点，停止遗留不可忽略。两组同低速覆盖期望59.1%、min_stride.06/max_air_credit1/完整交替脚步，仅long_stride权重1对6；PD/HD1910/STL不变。新critic/空Adam/LR1e-5/anchor5/normalizer冻结，两组各64×5smoke后1024×50，两组smoke/finite/normalizer/40Adam及导出均通过，control1024×50在途。GPU全11速度×3yaw×4×2独立seed227151/227157，结束看实际步幅/低速起步/逐例停止，勿重复启动或盲加轮。需后续CPU视频/原模型/扰动验收，无实机或部署，目标未完成。详见docs/walk-guided-training.md。

> **2026-10-16 v29起步状态诊断完成，无活跃训练：** 本地v28学生先运行8秒再v6模拟教师接管，4例切换前401帧qpos/velocity/obs/raw/target逐值相同，原停滞可启动，但前进速度持续波动。远端旧轨迹与本地不逐值相同，拒作配对，补跑本地基线。三反馈响应设置各4例；方向化新seed226171/226177与原教师各12/12存活，vx滚动误差.005272→.002821，yaw几乎不变，停止.003482→.003738；前进直行步幅约3.01→2.16cm，不能作为大步验收或最终示范，不继续拟合此小步目标。默认教师1200帧逐值保持，三响应配置NumPy/Torch各500帧通过。下一重点在低速摆腿时长/步频和停止保护，非盲目加轮；仅离线教师诊断，无学生更新/实机/部署。见docs/validation/20261016-walk-startup-handoff/。

> **2026-10-16 v28完成/未通过，无活跃训练：** job.exit0/source-check/完整归档0726cd8b下载核验。500更新低速GPU起步66→76/96，CPU8→9/12；纯低速直走仍有停滞，GPU速度误差改善19.56%未到20%门，CPU方向门失败。旧覆盖CPU跟踪改善且步幅10.48→10.43cm，但停止残速.00291→.00422m/s，GPU停止门也失败，未续1500。分阶段留出诊断表明起步拟合改善但GPU停止动作RMS退化8–9%，拟合不等于闭环起步；下一检查学生失败状态的示范覆盖和停止动作保护，不盲加轮、不放宽门。证据docs/validation/20261016-walk-startup；PD/实机不变，目标仍未完成。

> **2026-10-16 v28低速起步联合学习运行中：** 独立walk-startup-student-v28-20261016，launcher2541904/worker2542343已核验活跃，637源hash保持；v16step500初始化，联合旧v11CPU/v13GPU及v27低速示范，normalizer冻结/新Adam1e-4/stop8/transition1，仍单61→14导出。5更新冒烟finite/导出误差6.41e-7通过，500更新已完成并在CPU/GPU旧覆盖及低速启动分别筛选，勿重启。仅实际起步改善且旧跟踪/停止/步幅保持才续1500/3000；约3cm低速示范仅启动引导，不算最终大步验收。源代码通过本地代理推送，实机/PD/部署不改；目标未完成。详见docs/walk-guided-training.md。

> **2026-10-16 v26/v27已完成，无活跃训练：** v26原96/96、教师95/96，223151/world1[-.4,-.6]失败保留；低速48/48且vx/yaw偏差降至.00218/.00741，支持低速启动示范而非整套验收。v27 CPU30/30、45000帧/20训练10验证，GPU40/40、60000帧/30训练10验证、80运动阶段均启动，GPU seed224131/job.exit0/PID2477722已结束/源hash通过。两集SHA9eb0af11、dc731198下载finite/整世界划分核验。脚步仍约3cm，非最终大步目标；不重采或部署。下一从v16step500联合原v11CPU/v13GPU+新低速数据，分开检查旧覆盖与低速启动，学生尚未开训。详见docs/walk-guided-training.md，目标未完成，无实机/PD改动。

> **2026-10-16 v25已结束/未合格，v26冻结示范验证中：** v25 resume.exit0/source-check/归档cd0728c0通过，两组各1024×50/Adam400；GPU原/控制/大步263/264且同失败222151/world54低速-.05直行，无新增失败但速度略恶化。CPU三版各36/36，纯±.05都未迈步，大步组停止略退化，未续训。不把最小步幅奖励测试当已学会大步。v26扩展模拟教师纠偏.25/查询.6，默认数值保持、CPU/Torch宽/窄界限和停止旁路测试通过；v6原/教师低速±.05与高速±.4复合yaw，launcher2442489/评估2442892活跃、625源hash核验，勿重启或提前采示范。详见docs/walk-guided-training.md。无实机/部署/PD修改，目标未完成。

> **2026-10-16 v22失败/v24通过/v25覆盖与步幅对照运行：** v22 step500仍GPU停止失败，归档07d14eaf通过，不加轮。发现原奖励在.05m/s的2.75cm已饱和；新可选min_stride.06/max_air_credit1，默认数值2048组严格保持。两组共同宽速度±.05至±.4，只有步幅奖励不同；v24各64×5/Adam40/finite/导出/硬件合同测试通过，归档4df7bc10核验。v25从v16step500 actor各1024×50；control训练导出已完成，原launcher2359860资源门失败exit1，未重训。当前resume2378535/评估2378603活跃，先control-screen再stride train/screen，33命令×4×2新seed222151/222157。完成看resume.exit/source-check，勿因旧job.exit1重启。详见docs/walk-guided-training.md。无部署/实机，目标未完成。

> **2026-10-16 v21完成/v22过渡微调活跃：** 完整曲线每seed112事件原/候选均停稳；v16均停稳比v6慢.03–.04s，一秒后均速约.001m/s，原均值门仍失败。静止示范误差降低但GPU过渡误差未降，故独立walk-stop-transition-v22-20261016新增停止衰减首50帧损失×4，stopWeight8/同v11起点/其它门保持。launcher2266575/worker2266925活跃、624源hash核验；5更新导出冒烟通过，已到500筛选，勿重复启动。v21归档787dc04e下载核验，详见docs/walk-guided-training.md。低速大步/宽速度/连续切换/抗扰仍待完成，无实机部署，目标未完成。

> **2026-10-16 v19/v20完成，v21停止曲线诊断中：** v19独立回归无丢例，但v16 step500相对v6四项/相对v11两项停止退化，未验收。v20仅运动后的停止切v6，GPU均残速改善4%且无相对v16退化，但对v6两个[-.1,0]仍失败；CPU4组停止前轨迹逐帧一致、换停止策略未改善。两轮exit0/source-check/归档下载通过，不采组合示范或盲加轮。v21冻结v6/v16完整motion trace，launcher2227901/worker2227909活跃、625源hash核验，不重启；分析首秒行程/连续停稳时间/一秒后残速，原门保持。详见docs/walk-guided-training.md，无实机/部署/PD修改，目标未完成。

> **2026-10-16 v16已结束/v19冻结回归活跃：** v16 step500停步/步幅/CPU/GPU跟踪续训门通过，1500虽然跟踪再改善却5项停步退化，未续3000；job.exit0/source-check/完整归档1aad9fa2通过。保留step500 SHA457f2761做独立v19，新218151/218157分别对v6/v11长保持反向停步；launcher2158605/evaluator2158606活跃、625源hash通过，勿重复启动。CPU扰动小样本三版各12/12未倒；低速±.05仍不迈步。v17脉冲大步参考速度不平稳，拒绝训练；v18宽速度教师+.4可准确、-.4仍仅-.361，未采示范。详见docs/walk-guided-training.md。无实机/部署，目标未完成。

> **2026-10-16 v14结束/v16对照启动：** v14 step500完成/job.exit0/source-check/归档dd7830ff通过，GPUvx/yaw偏差改善14.5%/35.4%、CPU步幅10.4cm保持/停止均速改善，但GPU两停步工况退化，未加轮、不合格。分项验证停止动作RMS约+14%，故独立walk-joint-stop-v16-20261016从同v11起点，唯一改变停止示范损失权重1→8，预定门保持；launcher2098780/worker2099203活跃、621源hash核验，勿重复启动。低速v15扩大CPU教师界限可启动旧失败例但超调且低速仅约3cm步，非最终示范或运行端改动。详细见docs/walk-guided-training.md。无实机/部署/PD改动，目标未完成。

> **2026-10-16 v14联合微调已启动：** v11 step500 actor初始化/冻结normalizer/新Adam1e-4，CPU v11+GPU v13整世界留出示范各512等权拟合，先64环境构建/5监督更新导出冒烟，再500/1500/3000且CPU保持+GPU跟踪改善/逐方向停止/步幅门全过才追加。独立walk-joint-feedback-student-v14-20261016，launcher2046789/worker2047426已核验活跃，620源hash核验通过，尚无新模型结果。不得重复启动；详见docs/walk-guided-training.md。低速启动/高速偏航/扰动/连续切换待解决，不部署、不改PD，目标未完成。

> **2026-10-16 v11结束/v12参考/v13数据完成：** v11学生500步CPU独立yaw滚动误差.2165→.0564（约−74%），速度改善/步幅保持，无运行端反馈；GPU两停步工况失败，未续1500。job.exit/source-check/归档75b7d692通过。±.05仍不迈步，±.3/.4有偏航需补训练。GPU反馈参考v12原/教师120/120、vx/yaw均偏差.0134/.0765→.00107/.00671、平均停止略改善，但单原地转向停步+.0018m/s仍失败，不算验收。v13 GPU示范100/100存活/15万帧，75训练25验证，数据SHA02303bb9已下载核验；采集已结束，无活跃进程。下一步CPU+GPU纠偏示范联合蒸馏，再补低速启动/宽速度/抗扰动，不重采或直接部署。详见docs/walk-guided-training.md。

> **2026-10-16 v11反馈示范学习已启动：** v10示范持续vx/yaw准确但停止残速.0197m/s退化，已终止v10学生1867243自身进程组，保留step500/abort-reason。v11停步时旁路纠偏，同12例停止均残速.003785≈原.003851、持续±.2/±.6跟踪准确、脚步约10cm；不是已训练策略效果。新75组示范全存活/11.25万帧，整初态50训练25验证，数据SHA126eb80d…。GPU0独立walk-feedback-student-v11-20261016/launcher1907759：v6学生actor/冻结normalizer/Adam1e-4，加旧GPU状态动作锚点，500/1500/3000逐段CPU+GPU闭环与停步/步幅门才续，无特权反馈进入导出。当前已核验PID活跃，尚无v11学生结果，勿重复启动；详见docs/walk-guided-training.md。无部署/实机，目标未完成。

> **2026-10-16 v8/v9b完成且未合格：** 修复GPU BAM m6二次摩擦缺异号/等幅门，2205组×2精度×3摩擦比例CPU oracle通过；原/学生120/120，yaw均偏差.02816→.02743，非主要转向根因。IMU自由关节对照误差<1e-6；CPU每轮forward也未消除差异。v9b从v6学生actor-only、新critic/空Adam、冻结normalizer、LR1e-5/anchor5，长保持+加强平均跟踪/停步，64×5/导出通过，1024×150后无丢例但vx/yaw偏差恶化15%/15%、步幅+2.35%，已停止不加轮。CPU18/18不倒但转向未合格。v9旧optimizer覆盖LR的启动已终止，失败证据保留；v9b job.exit/source-check以实际记录为准。下一步反馈纠偏模拟示范再蒸馏，非运行端补偿；无部署或实机。详见docs/walk-guided-training.md。

> **2026-10-16 v7长保持诊断完成：** GPU随机/固定执行器/关闭列明DR三条件，原与3000步候选每条件各120/120存活，原瞬时yaw门仍失败；随机相位vx偏差.062→.013、yaw.216→.074，关闭DR .060→.007/.217→.028，但停止残速略增。CPU/GPU负转向差异仍存，编译脚接触/关节/求解器关键参数一致，15个servo碰撞体GPU禁用而CPU启用，代表CPU直走无非脚接触，根因未定。v7b与nominal-v7 job.exit0/脚本hash/归档通过，无活跃进程；初始v7路径错误保留。视频student-longhold.mp4已渲染查帧。下一步核对执行/实际运动，勿盲重训或部署；详见docs/walk-guided-training.md。

> **2026-10-16 v6已结束/持续转向仍未通过：** 3000次监督更新job.exit=0与source-check通过，完整结果归档SHA69e379ce核验；各阶段原筛选仍失败、无原成功丢失。新seed204101/204107 CPU原/候选24例30秒均完成，17–27秒直走输入±.2实际+.224/−.202m/s、脚步10–12cm，但yaw−.6实际−.91至−.95、+.6实际+.78至+.79，零转向仍偏航。不可继续盲拟合同批参考。已确认GPU delay在5ms物理子步，排除误按20ms策略步解释；下一步GPU长保持与固定执行器隔离，再修订参考/奖励。无活跃训练启动、无实机/部署，目标仍未完成。详见docs/walk-guided-training.md及docs/validation/20261016-walk-student/。

> **2026-10-15 v5参考完成/v6学生学习启动：** v5原/教师均120/120，平均相位速度偏差.06404→.01171m/s、yaw偏差.20962→.06324rad/s，0.2直走约9–10cm脚步；瞬时yaw原门仍失败、复合动作有偏差，不宣称最终通过。源hash/归档SHA通过。根据明确改善允许作为实验学习起点，失败项保留。独立walk-guided-student-v6-20261015/GPU0/launcher1471435已启动，252环境24秒参考采集、整环境留出，原actor/冻结normalizer，监督Adam1e-4按500/1500/3000段且实际误差改善才续；每段导出/闭环新seed筛选。导出仍单61→14网络，无教师表运行端映射。尚无学生结果，不部署/不改PD，无实机。详见docs/walk-guided-training.md；检查PID与student-progress/job.exit，勿重复启动。

> **2026-10-15 v4完成/v5二维参考筛选中：** v4原/教师均120/120不倒，相位平均速度偏差.06404→.01314m/s、yaw偏差.20962→.06854rad/s，0.2直走脚步约9.5cm；但瞬时yaw原门失败，后退转向速度不足/部分超调，不能宣称完成或部署。v4结果/源hash归档核验通过。v5对前向和yaw都按速度×转向二维表修正，端点/idle/连续/dict不变性测试通过；独立walk-guidance-v5-20261015/GPU0/launcher1431167只在冻结参考评估，未学生训练。191111/191117均为拟合seed，后续须独立回归。所有阶段原失败门保留，公开结果docs/validation/20261015-walk-guidance；详见docs/walk-guided-training.md。无主板操作/PD修改。

> **2026-10-15 v2结束/v3教师参考完成/v4参考修正中：** v2 duration至450轮改善vx4.17%/yaw2.87%/步幅4.98%、仍丢1例而停；stride150改善不足1%、丢1例而停。源hash/完整336MB结果归档核验通过，无v2训练进程。v3教师冻结动作参考120/120不倒，速度误差约减半、步幅约3.4→7.0cm，但yaw瞬时误差未降且后退直走偏航，未合格；不是已训练新模型。v4按GPU实测修正教师参考，seed191111/191117转为拟合数据，GPU0独立walk-guidance-v4-20261015/launcher1401673在评估，原基线缓存有源hash/初态校验；查实际PID/progress/job.exit，勿重启。端点/idle/连续性/dict不变性测试通过。尚未学生训练，不部署，不改PD/实机。详见docs/walk-guided-training.md。

> **2026-10-15 v2分段进度：** duration隔离/64×5/150轮通过，速度/yaw误差改善2.32%/1.92%、步幅+1.16%，仍丢原成功1例，仅符合探索续训门，已继续300；stride组待调度。CPU单初态原actor+.3/+yaw.4可产生.198m/s/偏转.026rad/s/约10.3cm步，作为待验证动作引导参考，不是输入标定或实机结论。launch1246825仍在运行，按其PID/子进程与stage结果跟踪，不重启。

> **2026-10-15 用户授权持续优化/分段加训（最新）：** 新目标为速度/转向跟踪与明显大步，允许有改善继续加训，无改善改奖励/策略或模拟动作指导。独立walk-progressive-v2-20261015/GPU1/launcher1246825：duration保持v1奖励延长训练，stride仅大步权重1→6并移除额外三重平方衰减，保留交替落地/时间加权。原actor/新critic/空Adam，LR3e-6、PD/HD1910/STL/物理/执行合同保持；各64×5后按150轮分段，最多600，实际筛选持续进步才续。最终仍要求无原成功退化，阶段10%改善不等于实际速度/大步目标已达成。勿重复启动；详见docs/walk-progressive-training.md，以实际PID与progress/job.exit为准。无实机访问或部署。

> **2026-10-15 速度/转向/大步幅首轮已结束（覆盖准备状态）：** walk-tracking-stride-v1-20261015完成64×40执行核验、64×5试训/导出及1024×50/Adam400。冻结normalizer、原actor L2保留10、LR3e-6、PD/HD1910/STL/kissqy标称物理保持。修复评估导入后仅补跑冻结对照，没有重训；原120/120、候选119/120，丢seed191057/world6。vx/yaw误差改善不足，平均脚步约3.5cm，未达到更大步幅/跟踪10%改善门，screenAccepted=false，不部署、不盲加轮，无活跃训练。源hash/结果归档SHA核验通过，完整检查点/日志已下载；候选5d39de15。详见docs/walk-tracking-stride-training.md和docs/validation/20261015-walk-tracking-stride/summary.json。无实机访问或动作，仿真非实机验收。

> **2026-10-15 速度/转向/大步幅微调准备已启动：** 用户授权继续行走微调、优先大步幅。独立walk-tracking-stride-v1-20261015/GPU1/launcher1124550；原实机actor恢复而非停止候选，kissqy标称物理/HD1910/当前STL/PD/50Hz保持。15种前后直走/复合转向/原地转向/静止命令，速度和yaw跟踪+完整交替大步奖励，固定LR3e-6、原actor on-policy L2保留权重10、冻结normalizer。配置/命令覆盖/步幅单调测试已通过，64×40执行合同目标误差2.676e-8/raw历史及速度0、恢复actor误差2.623e-6通过；64×5试训与正规导出通过，独立1024×50/Adam400已完成，ONNX误差2.861e-6/finite/normalizer保持；首评估仅因quat_apply导入路径失败，未重训，修复后独立评估PID1158681进行中。15×4×2新seed配对分方向筛选，失败不追加或部署。查看evidence/progress.json/job.exit，勿重复启动；详见docs/walk-tracking-stride-training.md。无实机访问或动作，尚无新模型效果结论。

> **2026-10-15 停止微调已完成/未通过（覆盖准备中记录）：** walk-stop-v1-20261015，1024×100/Adam800/正规导出/源hash通过，最终113/128较原128/128退化。冻结25/50/75筛选后25较佳，但独立seed152031/152041原127/128、25候选126/128，丢world54；停止改善6.7%/2.5%，未过门。CPU18例均不倒、停止残速−18.9%，但后退yaw漂移−.1757→−.2430rad/s恶化。deploymentAccepted=false，不部署/不盲加轮，PD未改，未访问实机。候选25SHA8c2481d6，最终45e92f43；检查点/日志/筛选/独立回归已下载，selection.json为最终状态。详见docs/walk-stop-training.md与artifacts/walk-stop-v1-20261015。

> **2026-10-15 用户授权停止稳定性微调（准备检查中）：** 独立walk-stop-v1-20261015/GPU1，当前launcher4160900，先tests/64世界40步合同检查与当前1cbf8730 ONNX恢复，再64×5正规导出冒烟，成功后1024×100，固定LR1e-5/std.05/冻结actor normalizer/新critic及Adam；模型不自动部署。12秒前后走停循环，新增停止平移/角速率与大倾角代价，保留原行走跟踪与头部目标，PD未改。显式kissqy标称物理/当前STL/HD1910/scale1/.5和.7滤波/50Hz，电压与摩擦随机化不是实测辨识。结束两个新seed各64环境原/候选停止与行走配对筛选。初次heading范围冲突在训练前退出，evidence-attempt1保留；检查progress/job.exit，勿重复启动。详见docs/walk-stop-training.md。

> **2026-10-15 17:43最新前后行走失稳与仿真回放：** 用户确认“就刚刚”后再次下载最新记录，17:43:27会话不同于先前17:39:59零指令站立失稳，不能混用。新包保存2400walk帧、4个连续600帧窗口，中间记录缺口未插值；前进最高+0.20m/s，最后12秒先原地yaw约−.596，随后前进主要+.17～.19，再出现后退，最后半秒约−.20。末600帧50.0006Hz/最大24.665ms，零缺反馈/校验，实机最大53.13°后倒地保护；运动关节P5D0，电压最低6.5V。各窗口从首帧关节/重力/角速率/速度与动作历史重建，重放已滤波指令和实测电压动作缩放、时序量化5ms；kissqy标称物理/BAM kp5/固定7.4V电机电源/固定20ms延迟/理想平地，未知根平移速度与齿隙状态置0。四段仿真均不倒，末段max5.06°，不能称整条路线连续精确复现或硬件响应一致；ONNX实测obs重推最大误差1.79e-7。未发动作/改PD/部署。新证据附件/舵机/2026-10-15/walk-fall-latest-recheck/analysis.json；仿真平级microduck-rl/artifacts/walk-fall-replay-1743/。旧17:39包另留walk-fall-1739，413帧全零指令/50.005Hz/实机57.92°，仿真4.08°，两次均未被标称仿真复现。

> **2026-10-15 用户授权 kissqy 仿真参数对齐：** 新增显式 `--physics-profile kissqy` 比较配置，允许仅该诊断使用供体0.82kg质量/惯量/装配，原HD1910/STL训练默认不变。17碰撞体编译几何验证、±1°齿隙、5ms×4/50Hz及求解器对齐；PD未改，BAM kp5仅代理。3项合同/拒绝漂移测试通过；18组±.2m/s与0/±.6yaw、3初态全部未达60°跌倒门，最大5.236°，但速度偏低和转向不对称仍在。固定20ms延迟/7.4V/平地为标称抽样，未复制随机地形或全部训练随机化。未访问实机/部署/训练。详见docs/kissqy-walk-physics.md，证据artifacts/kissqy-physics-20261015。

> **2026-10-15 行走叠加转向仿真（42例）：** 用户要求0.2m/s加方向。沿用当前1cbf8730 ONNX/本地STL机械限位/当前编码/50Hz/运行滤波/BAM kp5、7.4V、平地摩擦1。2秒起步±0.2m/s，4秒叠加yaw0/±0.2/±0.4/±0.6rad/s，9秒两指令归零，观察至12秒；2速度×7转向×3配对初态=42例，全部未达60°跌倒门，最大倾角前进5.22°/后退5.37°，无额外冲击。5–9秒实际body前后均速前进约0.097–0.105、后退约−0.080至−0.089m/s。手柄范围±0.2rad/s下，前进实际yaw+0.025/−0.205，后退+0.108/−0.087；零转向前进仍约−0.074rad/s漂转，不能只凭不倒称速度/方向跟踪准确。±0.4/±0.6是离线扩展工况，未修改实机手柄上限。已渲染后退左右并列回放、检查7秒帧，保持直立但方向响应不对称；结果不证明实机也能稳定。未连接/发实机动作/训练/部署。脚本/42组轨迹/summary/视频：microduck-rl/artifacts/walk-turn-20261015/。

> **2026-10-15 当前行走模型抗冲击仿真（222例，非实机阈值）：** 使用部署1cbf8730 ONNX、本地STL/机械限位、当前标定编码、50Hz/动作scale0.9/头腿滤波0.5/0.7、前一帧关节速度及8ms附加延迟；BAM kp5/7.4V为代理，不等同已辨识固件P5D0。平地摩擦1，前进/后退指令±0.2m/s，第2秒起步，第4/4.16/4.32秒沿当时身体水平朝向前后左右施加一次基座平移速度脉冲，7秒归零，观察至10秒或倾角≥60°。3配对初态×2速度×（无扰动+4方向×9强度0.1/0.2/0.3/0.4/0.5/0.6/0.8/1.0/1.2）共222例，基线6例均不倒，实际均速约前进0.096/后退−0.086m/s；所有施扰成功，无施扰前失败。首次观察跌倒：前进时向前0.8(2/3)、向后0.8(1/3)、向左0.6(1/3)、向右0.5(1/3)；后退时向前0.8(1/3)、向后0.5(1/3)、向左0.8(1/3)、向右0.5(2/3)，后退向右0.6为3/3倒。0.4及以下本轮全部不倒，不代表安全保证；3例计数非总体概率，Δv非推力/冲击持续力，未覆盖摩擦/固件/初始接触不确定性。代表后退向后0.6回放在施扰后约0.46秒超过60°，已检查施扰和末帧，视频截止保护门。无实机访问/动作/参数修改/训练。脚本、summary、阈值表、热图及视频位于microduck-rl/artifacts/walk-disturbance-20261015/。

> **2026-10-15 实机起身轨迹重建仿真（CPU诊断，非精确复现）：** 用户要求核验23:50姿态在仿真能否起来。使用已部署4b9c82e5 ONNX，逐帧实测观测重推输出最大误差4.768e-7；当前训练XML哈希不匹配，改用getup-mechanical-training-20261012/round1冻结副本，全部metadata机器人源及HD1910参数SHA检查通过。从实测0/1.8/3.8秒观测重建重力方向、关节位置/速度、角速率、raw/EMA历史，三种地面间隙0/3/6mm，共9例15秒CPU闭环均最后3秒稳定100%；起始姿态3秒达到运行完成门控，两中途姿态重启后约1.36–1.48秒稳定。首姿仿真约0.7秒后维持~1.6°，实机则两次回到~46°，显示仿真没有复现反复失稳。未知基座平移速度置0、yaw任意平地等价、接触/高度重建；使用BAM P5辨识代理而非已标定实机P5/D0响应，中途动作内部状态未完全重建，因此不能声称精确sim2real或9例等于成功率。未训练/改权重/改实机/发动作。脚本/9组轨迹/NPZ/summary与对比图位于平级microduck-rl/artifacts/real-getup-replay-20261015/。

> **2026-10-14 大幅舞蹈 v5 抗扰续训已启动：** 用户要求加强动作与抗干扰。v4 已完成800轮，正常64/64、训练式扰动55/64、周期压力57/64，尚未通过抗扰验收。独立 `dance10-resilient-v5-20261014`（launcher PID122660）载入v4 final，保留完整大幅动作且取消幅度渐增；训练水平推扰±0.18m/s、间隔1.5–4秒、每回合24秒连续三轮。HD1910/当前STL/61维/EMA/机械限位/45度终止门保持，actor归一化冻结、Adam重置、std .08、LR1e-5、seed121031。12项本地测试通过，64环境×5轮及带归一化ONNX导出通过，已进入1024环境×1000轮阶段。结束自动按v4原评估条件/两旧seed对照，再用120011、120017各64环境做新旧配对压力评估，核验初始qpos/qvel/观测哈希及实际动作幅度，并留视频复核；不宣称全部物理随机量一致。查看evidence/progress.json与job.exit，勿重复启动。未部署或访问主板，仿真非实机验收。详见docs/dance10-training.md。

> **2026-10-14 用户要求舞蹈更魔性/大幅v4启动准备：** v3训练及回放已完成：正常128/128，旧stress63/64，新seed62/64、63/64，仍有3失败/2旧成功退化，未完全验收或部署。用户观看后明确要求放大动作。新增独立Dance10Expressive，头偏航/pitch及身体侧摆2倍、歪头/屈膝2.5倍（下沉目标15mm），前300轮渐增/后500轮全幅；加强body跟踪、降低固定腿姿奖励，HD1910/STL/EMA/61维/终止门保持。11测试通过，目标头部加HOME距限位≥.05rad；小幅/全幅参考各32/32无倒地，实际头偏航30.43→60.63度、body roll5.37→12.05度、高度7.33→20.19mm，未当抗扰验收。64×5/导出已通过，800轮阶段在途。原v3权重固定作种子与小幅/全幅参考；64×5/导出通过才1024×800，结束两新seed三条件全幅回放及实际幅度/视频，禁止用小幅课程结果当大幅验收。独立dance10-expressive-v4-20261014 launcher3979552，首次上传超时仅空目录、重传后启动，无旧作业重启或实机操作。查看progress/job.exit勿重复启动。详见docs/dance10-training.md。

> **2026-10-14 舞蹈v2优化完成/v3恢复训练启动：** v2正常/训练扰动128/128仍通过，摆头/点头幅度相关性通过，头部误差降约30–40%；stress46→54/64，但丢原6例，accepted=false。100/200/300检查点44/51/55且丢原11/9/6，未解决退化。v2完整证据SHA557124a8381bd6dfda803e20cb29af4d4397a6daff6a0ddf419617b4f19349ea下载核验、源hash保持。独立v3 Dance10Recovery扩大初态/推扰范围，保持节拍幅度/HD1910/STL/EMA/奖励/终止角门，载入v2 final，actor normalizer冻结/Adam重置/std.08/LR1e-5，先64×5再1024×400及五组回归/两新seed100003、100019的原v1配对压力筛选。11测试通过；新服务器目录dance10-recovery-v3-20261014，launcher3434233，查看progress/job.exit勿重复启动。当前没有已通过的抗强扰动新候选，未部署/访问主板，仿真非实机验收。详见docs/dance10-training.md。

> **2026-10-14 舞蹈稳定性优化（进行中）：** 用户要求继续优化倒地。冻结v1 model999实际回放：nominal/robust×seed81371/91373各32例合计128/128完整16秒未倒，不能将探索训练日志当成策略失败率。诊断stress（初态roll±.08/pitch±.10、关节±.04，每3秒水平速度推扰±.16）仅46/64；18失败中17例在推扰后.7–1.2秒。新独立Dance10Stable保留节拍/幅度/HD1910/STL/EMA/观测/终止角门，修复zero-twist打滑惩罚失效、加超额倾角/偏航代价及恢复练习。actor/critic加载v1、Adam重置、actor normalizer冻结、std.10/LR3e-5；smoke64×5/导出通过，1024×400在途，launcher3319271位于新服务器目录dance10-stable-v2-20261014，不重复启动。10项本地测试通过，结束按固定seed/配置做新旧逐例存活和幅度回归；原评估未存完整reset物理状态，勿称初态逐数组已核验一致。未宣称修好，不自动部署/访问主板。详见docs/dance10-training.md及artifacts/dance10-optimize-20261014。

> **2026-10-14 十拍舞蹈首轮准备/启动：** 用户明确要求着手训练约10拍魔性鸭子舞，授权本次独立服务器训练。新增 Dance10-HD1910：100BPM/6秒舞蹈+前后各1秒站姿、61维指令观测、当前replica与HD1910/P5/EMA旧权重.45；从头训练、无官方教师。11项本地检查通过。服务器已迁移到 `/CodeRepo/Code/leiwu30/1_code/microduck-training`（已核对迁移记录），旧home路径失效；本轮独立 `dance10-v1-20261014` launcher2705983，32世界3秒站姿检查无重置/最大倾角6.2度、64×5冒烟及归一化ONNX导出通过，2048×1000已真实迭代（详见本地launch-status.json）。以 evidence/progress.json/job.exit 和日志为准，不重复启动；尚未动作验收，不自动续训或部署、不访问主板。详见 `docs/dance10-training.md`（本记录位于docs时使用同目录文件）。

> **2026-10-14 用户要求统一训练硬件（本地整理）：** 所有新训练仅使用 `robot/replica` 当前 STL/MJCF 与 `robot/hd1910/1910_m6.json`。`hardware_profile.py` 为参数入口，禁止恢复 XL330/P200/6V 下限、官方教师默认下载、供体质量/惯量/装配覆盖。旧任务名称现在也走 HD1910，历史证据和权重不改写。`hardware-inputs.json` 记录当前输入 SHA；变更几何须显式审计更新。当前模型内 `xl330.stl` 是几何名称，不等于执行器型号；质量/惯量未重新辨识。训练/预览/导出须保持同一硬件来源与各技能动作合同。仅整理与软件验证，不自动训练、访问主板、部署或推上游。详见 `docs/hardware-profile.md`；本条覆盖下文旧官方默认配置说明。

> **2026-10-13 16ms新候选关键配对改善/全场回归启动（最新覆盖）：** screen-v2 PID2284601/job0全部12例结束，初始qpos/首velocity逐值匹配、270源hash保持、完整SHA6cd4150cc82e31569c507f4cfa91c9c6253ed28471568ea60829b0dd8b1bca6d下载核验解压及delay16-screen-v2-final-audit保存。新Delay16在14/16ms各6例中都无baseline旧生存/有效前进丢失，.2关键pile6两trial两延迟均存活前进，均速.1142～.1229m/s；flat .15亦前进.0576～.0750。相对encoder20在16ms/pile3/.02/240017/1/.25仍丢1生存和有效前进（baseline该例原失败），新候选11/12生存，不能称全面合格；样本全属拟合内关键筛选，未实物验收。旧screen缺metadata失败完整SHAf8a4fa33、准备完整SHA7326d339均下载核验/源hash保持，失败源和日志保留，不重训。
>
> 新独立CPU walk-delay16-fullscreen-20261013 PID2324408已启动，Baseline/Delay16×14/16ms×三地面×四速度.1/.15/.2/.25×seeds130031/240017×2trial共192 actor例（96配对）；复用已完成66例，只新增126例，不重复GPU训练或抓实机。所有policy逐例初态/首velocity必须匹配，保存目标frame p95/max、存活/实际前进及丢原成功集；完整回归结束前无合格/待部署结论，后续须真正未用seed与正式验收。参考已完成baseline16/旧delay14/关键candidate trace，复用原始trace SHA记录并连同新全帧归档；跨地面初态受静置影响，非实测地毯剪切/全CPU-GPU动力学证明。追踪launch/progress/job.exit/comparison，不重复2324408。当前无活跃GPU训练、仅CPU回归；用户设备已关机，不再尝试主板SSH或要求开机。现役walk1cbf8730/P5D0及所有其它模型/标定EEPROM/增益/BLE/网页masked保持，无新权重部署/实机运动。七记录同步；仿真非实机验收，RL只本地提交不推上游。

> **2026-10-13 延迟配对筛选入口修复（最新当前阶段）：** 首screen2264708/job1仅因新ONNX旁缺Delay16.metadata.json在初例物理运行前拒绝，未重训或判模型失败；原完整归档SHAf8a4fa33保留。补配套原导出metadata的新独立walk-delay16-screen-v2-20261013 PID2284601运行12例，training=false/optimizer0/不接硬件，已完成14ms平地及pile3并推进pile6；后续16ms同三条件。所有源在launch SHA锁定，旧screen源码不改，不重复已完成GPU训练2251965。当前无合格/待部署候选，最终效果以完整comparison及逐例初態审计为准；旧“screen2264708在途”为历史。设备仍关机，仿真非实机验收。

> **2026-10-13 固定16ms行走对照训练结束/配对筛选中（最新覆盖）：** walk-delay16-transfer-20261013 launcher2251965/job0已结束，无活跃GPU训练；Surface仅固定lag7→8，seed68131且初始actor/critic/空Adam与encoder20原seed字节相同，其他训练/运行合同保持。64×5 smoke/1024×20主run实际Adam40/160、rollout7680/491520、actor normalizer逐值保持/finite/正规导出通过，Torch-ONNX误差3.815e-6/2.146e-6rad；主权重SHA485722b8…尚无效果结论accepted=false。完整归档SHA68fb5fe0722ba379e65c44a692d2da529271f7d7f0ddeb262b6479024738fd8a已下载核验解压，236 sourceHashes保持、delay16-transfer-final-audit保存。
>
> 独立CPU walk-delay16-screen-20261013 PID2264708已启动，等待既有训练结束后才复制唯一主ONNX，三关键条件各2trial×14/16ms共12例，不重复训练。baseline/encoder引用两延迟已完成源trace，逐例初始qpos/首velocity必须匹配；raw允许新actor改变，不假称raw相同。检查存活与有效前进（max .02m/s或指令30%诊断阈值），拟合内筛选非正式/实物验收；完成后下载全包SHA/sourcehash及comparison再作判断，勿重复启动。用户设备已关机，不访问主板或要求开机，现役walk1cbf8730/P5D0及全部其它技能、标定EEPROM增益BLE网页masked保持，无新模型部署或实机运动。七记录同步，仿真非实机验收。

> **2026-10-13 行走编码量化与延迟对照（最新覆盖）：** 用户确认设备已关机/App不能连接，继续离线，不再访问主板；现役walk1cbf8730/P5D0及全部模型/标定EEPROM/独立增益/BLE/网页/masked服务保持，无实机动作或新权重部署。旧GPU冻结1930331已完成，完整SHAdb832b8e核验：rough .2三actor baseline/control/surface生存25/25/24、有效前进24/24/23（32各），.25生存21/21/20、前进20/20/19；禁附加噪声仍有继承DR，非CPU同初态/全动力学等价。编码量化transfer2100330完成64×5/1024×20、Adam40/160、normalizer/sourcehash/正规导出通过，SHAe735b9cd；关键6例2136542完成，恢复旧.25生存但丢flat240017/0/.15有效前进及pile6/.04 130031/1/.2改善（14.92秒倒），候选拒绝。完整probe SHA ff380ad211007990a3fa6d0e4706e50c4054a9776c3f8433936fb1c7308509fd下载核验解压、266源hash/job0，encoder-probe-final-audit保存，勿重复/加轮。
>
> 冻结CPU delay probe2165698全部18例完成，14ms相对16ms baseline gain/lose生存1/1、有效前进1/1；Surface生存gain1/lose0、前进无增减；Encoder生存gain1/lose2、前进gain1/lose2。逐例qpos/首velocity/首raw相同；2ms会翻转个别成败，不证明实机UART/PD延迟根因。完整SHA1e853d4195e83c7482cc1017a4d417a2800aca9c32981ec474e007e3c9452c99下载核验解压/267源hash/job0，delay-probe-final-audit保存。读取安装版mjlab env每2ms write_data_to_sim→entity apply_delay→BAM compute；真实Actuator.apply_delay/DelayBuffer冻结4×14目标阶跃测试7/8tick在14/16ms首次进入控制，BAM compute_control/compute_torque无JSON command_delay二次叠加，保存安装源码及delay-implementation-audit。此是软件执行时刻核对，不是实机延迟辨识或全动力学证明。
>
> 新独立walk-delay16-prepare-20261013 PID2243674/job0，5配置/延迟检查与64×40两root算术通过：target2.8498e-8/history0/previousvelocity0/encoder position1.4901e-8；源hash保持，准备没有optimizer更新。首setup因提前import缓存安装tasks而ModuleNotFoundError，发生在GPU准备前、未训练；已有目录只审计恢复，不重建/重启；PYTHONPATH进程前设置后通过。新walk-delay16-transfer-20261013 launcher2251965已启动，seed68131，仅Surface分支固定lag7→8（14→16ms），不混入随机延迟分布变化。保留actor编码量化、24step/2epoch/4mini、奖励/探索/增益/限位/EMA/HOME/物理；直接复用encoder20原seed字节相同actor/critic/空Adam，非供体历史完整resume。每stage授权GPU0/1容量>12GiB/util<30，本run指定GPU1；先64×5/Adam40正规导出finite/normalizer，成功才1024×20/Adam160。当前尚无新候选效果/合格结论，跟踪evidence/launch/progress/job.exit不要重复启动。完成后同时14/16ms冻结配对保原生存及有效前进，过门才真正未用seed回归/正式验收，仿真非实机验收。RL只本地提交不推上游。证据RL artifacts/walk-fast-audit-20261013及scripts/server/prepare_walk_delay_control.py、launch_walk_delay_control.py。

> **2026-10-13 保留修复失败与编码器量化补齐（最新）：** update-fraction18例全部完成/sourcehash通过，三幅度均不合格：.25保旧成功但丢pile6/.04 .2改善；.5亦丢flat .15且丢改善；.75保改善但仍丢flat .15与pile3/.02 .25。完整SHA16faf74e下载核验，update-fraction-final-audit.json。新CPU单actor监督success-retention PID2034652完成/job0，anchor原两丢成功轨迹2500帧、Surface成功轨迹抽6000（seed171337）源trace SHA锁定，teacher在全部数据原输出复算误差0；固定LR1e-5/actor100 Adam步/原normalizer图保持，不更新critic、不改运行端动作scale或双actor切换。20/50/100正规单ONNX finite及Torch数值通过，源hash保持，完整SHAa2b1a38f下载核验。anchor/preserve实梯度step20=.093/.101、cos−.958，50=.095/.091、cos−.996，100=.087/.085、cos−.998，表明本批保留目标相抵，不证明唯一失效原因。CPU关键18例PID2043352完成：Step20仍丢两个旧成功，Step50/100丢flat .15且丢pile6 .2改善；均拒绝不部署或加步，完整SHA6571a27f下载核验及success-retention-probe-final-audit保存，所有seed仍属拟合诊断非独立验收。
>
> 审代码确认训练actor此前用连续joint_pos_rel_backlash，而native/CPU用4096步量化位置；此前审计只核target/history/velocity，不能称所有观测处理已对齐。新增encoder_feedback_position在齿隙合角及训练校准bias后编码量化，不按目标范围clip；critic仍保留privileged连续位置、偏置/噪声/IMU misalignment DR保持。新walk-encoder-prepare-20261013 PID2079869/GPU0/两root64×40通过，target2.85e-8/history0/previousvelocity0/position1.49e-8rad，actor恢复及源hash保持。全部actor/critic起始state与旧20轮seed逐值相同，配置规范路径后唯一差异actor joint_pos函数，非重新调奖励/探索/增益/限位；准备完整SHAc88e3732下载源hash核验，encoder-prepare-final-audit保存，不证明量化就是实机摔倒根因或全动力学等价。独立walk-encoder-transfer-20261013 launcher2100330已启动，GPU0每阶段容量门/seed68131，仅Surface量化分支（避免重跑旧Control）；先64×5smoke正规导出normalizer/Adam40后1024×20/Adam160，与旧Surface20单项处理对照，尚无效果结论不可重复。旧GPU32world冻结1930331仍在途、与CPU不同DR/初态上下文，.2 rough Baseline25/Control25/Surface24存活（32各），不能称Surface已全面合格或CPU/GPU根因证明。板上现役walk及所有模型/独立增益/EEPROM标定BLE/网页masked服务保持，无主板运动/部署；RL仅本地提交，仿真非实机验收。
> **2026-10-13 行走配对初筛全部结束（新结果，未通过）：** walk-surface-screen-20261013 PID1902317/job0，三actor×三地面×16例=144配对样本，所有逐例初始qpos及速度逐值相同、源hash保持；完整含分析归档SHA9505fa32下载解压核验，surface-screen-final-audit.json保存。Baseline/Control/Surface生存42/40/46（48各），有效前进21/16/31（诊断阈值max .02m/s或指令30%，非实机验收）；Surface .2共12/12存活且有效前进，baseline11/12，修复pile6/.04 seed130031/trial1。但是Surface新增pile3/.02 seed240017/trial0/.25倒下21.58秒，并丢flat seed240017/trial0/.15有效前进，不能因总分增加部署；Control新增4原生存例失败、丢9原有效前进。候选仍accepted=false，不扩大正式验收或加轮。GPU冻结PID1930331仍在途，.2Control在该上下文32/32而CPU平地4例停滞；上下文含继承encoder bias/IMU misalignment等，enable_corruption=False只关附加观测噪声，不能当同物理初态CPU/GPU差异根因证明。
>
> 下一新独立walk-update-fraction-20261013 PID1994939，仅CPU冻结MLP参数更新幅度.25/.5/.75三点×两退化场景与一改善场景×两trial=18例，不训练/不改原normalizer/不改运行时动作尺度.9、滤波/反馈。相同ONNX图仅mlp权重从原到Surface线性插值，端点纯推理复算及normalizer逐值一致/finite已核对，accepted=false。目的是检验更新幅度是否存在保留原成功又保留.2改善的区间，不能称降动作幅度或新PPO训练。reference锁定原screen逐例初态/结果及trace SHA，完成后核对同初态并评估；只通过关键例才全场地回归及真正未用seed。首scp超时导致准备脚本缺失被拒，无launch/训练，重传成功后才启动1994939，禁止重复。板上现役/所有标定EEPROM增益BLE/技能网页masked服务保持，无实物动作/部署。仿真非实机验收。
> **2026-10-13 行走20轮迁移完成/冻结筛选在途（最新覆盖）：** walk-surface-control-20261013 job.exit0，Control与Surface两分支smoke64×5/主run1024×20全部完成，Adam40/160、正规ONNX导出finite及actor normalizer逐值保持通过，完整归档SHA6b214436下载解压/全部sourceHashes核验，surface-training-final-audit.json保存。主run Control SHA62e10dba/Surface3d111856，accepted=false未部署。CPU独立walk-surface-screen-20261013 PID1902317已启动，三地面flat/合成pile3 contact.02/pile6 contact.04×四速度.1/.15/.2/.25×两seed130031/240017×两trial×三actor=144例，已有两个同源baseline地面32例复用（源代码逐项核验），只新增112例，不重跑旧baseline；所有模型同场地逐例起始qpos/速度核对后才比较有效前进/生存/目标帧及接触，跨地面最终初态受静置接触影响。已出现Control平地.2停滞/.25倒退，故该分支不能部署，Surface尚在筛选，未预判效果。独立walk-surface-frozen-gpu-20261013 PID1930331/GPU1容量门通过，32world冻结三个actor/两地面/三速度上下文诊断，seed97133/禁观察噪声/只加载actor不加载Adam，无优化器更新；目的是区分训练环境表现与CPU回归差异，不是CPU/GPU全动力学等价、历史训练重放或实物验收。所有候选无正式通过，板上walk1cbf8730及P5D0/其它模型设置保持，未访问主板或发运动。
> **2026-10-13 行走terrain迁移小预算训练已启动（最新）：** prepare-v5两root64×40全部通过/exit0/sourcehash保持/无overflow：目标误差2.8498e-8rad、raw历史及上一20ms反馈速度误差0，恢复actor与现役ONNX最大差2.6226e-6rad，critic76维；准备含全部失败case归档SHAed016e3d完整下载解压并逐项sourceHashes核验。独立walk-surface-control-20261013 launcher1873045已启动不可重复，seed68131；Control/Surface均64×5smoke正规导出/finite/actor normalizer逐值保持及Adam40通过，Control1024×20完成，Surface1024×20在途；每阶段GPU0/1容量门≥12GiB且util<30，未干扰其它任务。两个分支仅terrain不同，当前heightfield网格.04m，波长/高度为训练代理而非实测地毯。seed源SHAaeb4fc94已核验，新critic及空Adam非历史checkpoint续训，24step/2epochs/4mini/fixed LR1e-5/std.1、actor normalizer冻结而critic normalizer学习。最终Adam须160，候选accepted=false；训练完成后以冻结CPU相同场地/初态逐例对照现役及Control，检查有效前进（明确诊断阈值max .02m/s或指令30%）、倾角/接触/目标帧变化，不以存活冒充走路通过，只有改善且无原成功退化才独立seed扩大。当前未部署/未发实机运动，不能称快走已修。准备失败v2/v3/v4保留，无重复学习。RL只本地提交，仿真非实机验收。
> **2026-10-13 地毯/软垫行走诊断与迁移准备（最新覆盖）：** 用户确认50%直线前进在地毯或软垫摔倒，对应0.2m/s指令。现役walk1cbf8730/P5D0保持，本轮无主板运动或模型部署。CPU冻结扫描v1至v4、soft及pile均已结束且完整证据下载SHA/源hash核验；v1/v2/v3含旧10ms速度采样假设，v2低压未补动作电压缩放，不与v4的上一20ms反馈/标量电压补偿混作验收。正确walk供体惯量、±1°齿隙编码、DOF摩擦修订dof-id-friction-force-v1保持，非实机PD辨识或CPU/GPU全动力学等价。平地存活可能停滞/倒退；软接触部分0.2实际速度仅.008m/s，不能当走路通过。合成6mm起伏/contact .04的0.2在13.84秒持续前进失稳；.02中某初态0.4秒开走前倒下另作站立失败，22.36秒失败在减速后，不混为前进失败。高度场非实测地毯形貌/剪切，跨地面静置后初态受接触影响。真实交接已知观测重建12例归档SHA64c4ddde完整下载及源hash核验：四地面0.2均存活、实际均速.069～.124m/s，0.15均基本不走，0.25两例17.50/21.10秒失败；根线速度/齿隙初态/支撑高度未知，非现场完整重现。前600实测纯推理复算raw误差2.086e-7rad、filter/target/history0；缺失摔前8.013秒仍无法恢复。
>
> 新Control/Surface迁移准备无任何optimizer更新：seed68131，从现役ONNX恢复actor/烘焙normalizer，count合成1000000，critic新随机/Adam空，非供体完整checkpoint续训。两分支仅terrain生成器不同，共同奖励为当前Velocity transfer来源而非原donor历史完全重放。v2准备因继承观察延迟叠加显式previous缓存而速度误差2.77rad/s失败；修去继承delay后v3目标2.85e-8/raw历史0/速度0，随后std_type=vector不支持导致失败，修为库中直接逐动作14维参数scalar。v4 PID1846022恢复actor通过后Surface出现height field collision overflow，按验证/proc的独立进程组精确终止，prepare.exit143，不采用此地面审计。所有失败源/日志保留；新独立walk-surface-prepare-v5-20261013 PID1861691/GPU1容量门通过，heightfield间距.01→.04m并新增overflow拒绝门，64×40两root算术/恢复审计在途，不重启或称训练开始。审计通过才smoke64×5正规导出/normalizer/finite/Adam40，再1024×20两分支Adam160，保留实际前进及平地低速逐例回归，不盲加轮。训练前每阶段仅授权GPU0/1空闲容量，不干扰其它任务；RL只本地提交不推上游。详见RL artifacts/walk-fast-audit-20261013/runtime-scan及diagnostics；仿真非实机验收，旧失败保护/标定EEPROM/所有技能独立增益BLE/网页/masked服务保持。
> **2026-10-12 翻滚20轮微调已结束、未提速不部署（最新覆盖）：** case roulade-mechanical-transfer-20261012，无活跃训练/候选；GPU0/1容量保护、实际训练GPU1，不动他人。修接触匹配容量64→256后4测试、64×40目标1.191e-7rad/raw历史0、两分支64×5smoke及1024×20/同seed62173，Adam100/400、4正规导出finite/normalizer、231源hash保持，无有效阶段overflow。原actor8fdaf12c恢复，critic新随机/count合成1000000明确是迁移假设，非完整原checkpoint resume。仅未完成翻滚时间代价0→−.5，现限位/无EMA/编码/Feetech BAM/具名惯量及冲击平滑约束保持。首接触overflow控制权重废弃，全部失败源/日志保留；末wrapper重复trainingSeed导致job.exit1发生在训练/筛选/源核验之后，独立final-audit确认全部完成，不重训。
> 3秒稳定退出支持角门原207/256→Repair208/256，逐例丢42；但查看配对画面并几何交叉核对发现很多只是接触加权角328°没过330°而实际完整旋转364°、头顶latch且直立，不将门槛丢例全说成摔倒。独立几何交叉核对原255/256、Control250/256、Repair254/256，Repair丢原2例；均无1秒完成，平均1.307→1.327秒、共同成功配对慢1.60%，相对control仅快.39%，无有效提速。该交叉诊断未改变原验收门、不是正式或实机验收；候选拒绝，不扩轮/换板。当前板1秒交回walk与模拟约1.3秒完成有提前交接风险，但实机带载掉压/跟随也未完全辨识，不只延时便宣称已修好。完整147MB训练源码/所有checkpoint/轨迹/失败源/日志SHA006f2a9a…下载核验解压，几何/判据补充分别SHA84d3f52c/4c99603e单独核验，明确不属原归档成员；RL artifacts/roulade-transfer-20261012/{diagnostics,complete/final-audit,orientation-cross-check}.json。主板/其它模型/配置/App不改，无代发动作。

> **2026-10-12 用户授权翻滚微调（运行中，无实机动作/部署）：** 独立roulade-mechanical-transfer-20261012 launcher-v4 3100009（前三次已退出），GPU0/1容量守护实际准备GPU1，4配置检查通过；64×40无EMA/HOME/现行clip/4096编码目标误差1.191e-7rad、raw历史0；当前8fdaf12c actor+normalizer恢复误差4.768e-6rad。原critic/count不可得，新随机critic/空Adam与合成count1000000明确记录，非原checkpoint完整resume。Control/Repair共同机械止挡与目标裁剪、供体具名0.82kg惯量/Feetech BAM/P5/观测delay和标量掉压近似，两实机起始状态仅reset覆盖；仅未完成前滚时间代价0→−0.5，保留接触/头顶/矢状面与冲击平滑保护。各64×5smoke/导出后同seed62173、1024×20短程对照，完整前滚>=330度/头顶latch/直立落地200ms，独立256例配对比较1/2秒完成及冲击/目标帧代理；普通恢复直立不算完整翻滚。首GPU大批出现接触匹配overflow，明确废弃其训练结果并只停止本case；容量64→256及日志拒溢出校验加入，复用原seed，准备及基线导出通过，尚无有效训练改善/合格结论，当前板权重/1秒入口不变。证据平级RL artifacts/roulade-transfer-20261012/diagnostics.md，仿真非实机验收。

> **2026-10-12 坐站训练与执行链最终状态（覆盖此前运行中状态）：** Pitch及FixedLR两轮均结束，无活跃训练/合格候选。FixedLR各64×5冒烟及1024×20完成，5配置检查、四正规导出finite/normalizer/Adam100和400/全checkpoint固定学习率1e-5核验通过。初筛Control正常/扰动各8/8无磕头，但新seed677031/789017正常15/16、扰动13/16，相对基线15/16丢sit_cycle677031/0、/3，扰动仍3例接地、最大目标帧50.80→61.26度；拒绝扩大验收、部署或盲加轮。完整FixedLR归档SHA cf9c99bff413d5694b6a019dcf3f66b3f8bb8d9636ae1ed1678436adcae4ee08已下载解压、源hash逐项核验。真实观测审计为归档之后的独立补充，不冒称归档成员。
>
> 只读真实会话2026-10-12 15:21:28：新机械起身→walk→sit313帧→stand50帧→walk，坐站仍1c94934c。站起倾角5.02→34.50度、峰38.20度，颈实际17.40→−78.22度、末请求−90度、12帧裁剪，P5/D20、电压7.8V；原模型同真实观测复算误差4.768e-7rad，裁剪前颈目标最低−470.04度。FixedControl同旧观测仍−471.77度/12帧，仅反事实输出，不是新闭环实物响应。倾倒及低头已在切回行走前出现，不能判切换或单颈PD为唯一原因。
>
> 已确认并修复Rust执行链缺陷：新机械起身从legacy worker100启动，后续切到donor walk/sit/stand未进入native同步50Hz循环。现在切换清旧worker、保留历史且不重复观察当前采样，提升到native循环；native状态明确kissqy-sync50-v1/feedbackTargetHz50。134项Rust检查通过、1外部忽略，ARM二进制SHA3164689e1a12e8522b96c0ff5d00ef8615108883fcba6e28e42e4c3467a96961。主板空闲15卸力下仅二进制部署runtime-20261012-164043，旧getup-mechanical-20261012-v2保留回滚；模型/网页/标定EEPROM/独立增益/IMU/BLE身份保持，tofd/rsyslog仍masked。候选1000/1000完整只读关节反馈及20生产快照通过，无代发实机动作。新起身4b9c82e5和原坐站1c949权重未换；软件测试及只读部署不证明磕头已解决。
>
> 下一步需用户现场托稳做一次倒地起身→坐下→站起→停止/卸力采集修复后的执行链，顶住立即急停；基于新链的真实初态、raw历史/越界输出诊断再决定小预算训练，不把失败raw当正确示范。RL仅本地提交，不推上游；仿真非实机验收。证据：RL artifacts/sit-pitch-fixedlr-20261012/{final,real-observation-audit.json,board-chain-summary.json}；Observer附件/平台/2026-10-12/{sit-engine-handoff,runtime-20261012-164043}。

> **2026-10-12 坐站前倾预防20轮已完成，固定学习率对照已启动（最新）：** 原Pitch20 job0、4检查及4正规导出finite/normalizer/Adam100/400通过，当前actor无EMA机械范围/raw历史保持。原baseline同native sensor/1s→walk正常8/8无头接地，扰动8/8有2例20帧头接地；Repair20正常/扰动均8/8无接地，无baseline成功退化，多余行程少7.14%/20.30%，但正常右膝target帧p95 8.4375→8.9692度增加6.30%，拒绝替换。32例capture逐qpos/result匹配原screen，targets非实际运动。原main checkpoint5/10（实际6/11轮）正规导出含normalizer/Torch-ORT审计通过，均两条件8/8，扰动仍1例接地，膝p95不劣但左膝stress最大帧53.0859/54.7559>baseline52.6465度，亦未合格；不扩大正式验收/部署。源manifest/原handoff两个修改文件独立原源码snapshot哈希均核验；完整case归档SHA642c56062936b3b4fe1af1673a177822665332ccdd6a609dd27412ff0e2e5ff4下载SHA核验，final-audit/frame-audit/checkpoint-comparison保存。本轮早/末模型仅小样本诊断，未完整重现实物PD/P6行走对应BAM，不能宣传实际不磕头。checkpoint分析首次MLPModel原地pop复用cfg第二模型KeyError，修deepcopy、保留源码日志，没有重训/重跑动力学。

> 新证据checkpoint optimizer LR曾1e-5→7.59375e-5，不是全程小学习率/唯一原因证明。已启动独立sit-pitch-fixedlr-20261012 PID1394066，5测试通过，当前跟踪evidence/fixedlr-control；保留同teacher seed/critic-count借用/std.05/seed62437/奖励/20轮预算，只把adaptive→fixed1e-5，两组仍Contact−1及额外Pitch0/−2，先64×5smoke/正规导出/Adam100/lr逐组1e-5校验，再1024×20/Adam400及native-sensor1s→walk同初态正常扰动各8回放。GPU0容量门≥12GiB/free+util<30、驻留任务保持，不动他人/GPU1；当前未完成/无新合格结论，不重复启动。主板/倒地起身模型4b9c82e5/标定EEPROM/所有独立增益/BLE身份及masked服务保持，无实机动作或部署。RL只本地提交不推上游。

> **2026-10-12 坐站完整链与前倾预防对照（新进度，无实机动作）：** 用户再次授权继续训练。独立sit-chain-handoff-20261012 CPU原始陀螺/上一采样关节速度/无EMA/raw历史与现行sit→walk filter对齐，持续坐站、1秒及3秒切行走各正常8/8、扰动6/8或8/8，但扰动三组均2例头接地；首接地站起0.90～1.04秒，部分在切换前，不能说唯一切换根因。BAM P5保持，实机行走P6尚无物理映射，不声称完整实机链一致。旧滤波CPU结果为历史诊断，不混配成功率。新Pitch Control/Repair各64×5smoke与1024×20已结束/job0，4配置检查通过，trainseed62437，Adam100/400，normalizer正规导出及finite通过；共同当前ONNX anchor/借用历史critic-count/std.05/lr1e-5/无EMA/机械范围，均保留接地惩罚−1，仅Repair新增站起前倾>15°平方成本−2（坐下免罚），非效果保证。GPU0已有驻留任务保留、容量门检查，不动GPU1/其它任务。PID1280390结束，不重启：launcher先spawn后错误source路径未写manifest，已按/proc精确恢复原PID登记；修正未来launcher先hash再spawn，原源码保留。registry首次误复制本地包含未随case带入模块的注册表导致4检查拒绝，恢复独立原registry追加Pitch后4通过，首失败日志保留。初筛两条件baseline8/8、8/8，control8/8、7/8丢stand_cycle817031/1，repair8/8、8/8无丢成功且头接地2→0；多余行程normal63.915→59.352/stress144.540→115.199，但normal左膝最大帧27.861→28.389度，尚未合格。CPU完整逐帧重放捕获正在运行，核对与原qpos/result逐值相同后再统计p95/最大帧；不预设扩大或部署。前轮HeadControl完整归档SHAc67a265bd8b5a976e3178ad4eb13045b27e01521338cb450f3ea4166a5cd5a6e已下载，4导出audit核验，control-final-audit.json；训练证据完整下载状态覆盖此前未下载。原手动/保护/标定EEPROM/增益BLE/起身权重网页保持，无板访问/动作；RL仅本地提交，不推上游。

> **2026-10-12 坐站20轮对照已结束（最新覆盖运行中状态，未部署）：** sit-head-clearance-20261012 job.exit0/progress complete/accepted=false，控制与头颈接地代价修复分支各1024环境20轮。只统计sit→stand边缘后6秒，排除初始站立段冲击：基线正常8/8、扰动6/8且2例头颈接地；控制8/8、7/8且1例接地；修复8/8、6/8且2例接地。修复未改善扰动成功或磕头，相对控制少1成功，不部署、不盲加轮。sourceHashes及逐例初始qpos核验通过；该小样本非正式验收，未模拟主板站起1秒切换行走，未证明完整实机原因。compare-v2修正旧统计包含初始站立冲击的问题，comparison-v2保存本地；完整本轮训练证据尚未下载，probe完整归档已下载状态不变。App0.9.44已纠正仅移除中间坐站入口、右侧保留；板模型和所有设备设置保持，无实机动作。

> **2026-10-12 坐姿起身磕头优化（当前，无实机动作）：** 用户确认磕头来自坐姿站起，当前倒地起身4b9c82e5已用户成功反馈，保持板权重。App0.9.43已OTA并取消坐下预备/站起，只用倒地起身启动。observer代码/新模型cfe374a、App463f730已推送；RL4a1d7bf仅本地提交，用户明确不建远端、不推上游。当前坐站1c94934c无EMA/现行机械范围CPU新回放正常8/8无头颈接地；扰动+1帧速度延迟6/8，失败两例出现头颈接地（模拟法向力峰55.78/54.25N，不是实机力测量），其中一例首秒倾32.08°且接地；回放连续坐站，没有重放板1秒回行走，非完整实机因果证明。原训练允许头辅助，不等于其唯一根因。完整16例CPU归档SHAe376b5e767f59ce79e36cc9fa081371fb8ec1947de131d68ba25b3e33bcdd9cc下载核验解压，RL artifacts/sit-head-clearance-20261012/probe。

> 独立sit-head-clearance-20261012新小预算对照launcher1036540（v3）运行中，仅GPU0，GPU1他人高占用不动；两个分支共用当前actor映射/机械止挡+HOME+编码/noEMA/raw历史/Feetech BAM/观测/critic，区别为站起指令下头颈接地软代价0或−1。种子由当前ONNX恢复权重及归一化mean/std，128冻结观测Torch/ONNX差2.146e-6rad；critic及normalizer count借自历史HD1910坐站checkpoint，并非供体原critic/count，不能把此假设当已校准；两组共同探索std.05/lr初值1e-5/adaptive调度/trainseed61723/5epochs×4minibatches。2配置/代价检查通过，先各64×5smoke正规导出/finite/normalizer/Adam步100，后从同seed分别1024×20/Adam400，再相同8正常+8扰动循环；不是正式验收、候选不自动部署。首驱动误假设每轮8更新，实际20，已完成Control smoke复用不重训；第二次导出审计主进程误载全局任务注册表，改为显式独立src，源/日志保留attempt1/2，当前v3不重复旧训练。旧job.exit1属于前次退出，跟踪launch-v3 PID及当前progress，不只读旧退出标记；后续最终检查sourcehash、完整相同初态/成功保留/头接地/过程/frame再决定扩大。主板/标定EEPROM/PD/BLE/其它技能及网页不改，仿真非实机验收。

> **2026-10-12 起身实机反馈与完整仿真（最新覆盖）：** 用户确认当前新模型倒地起身已经站起来，此为一次用户实机成功反馈，不推广为实机成功率。4800例正式仿真已结束/job.exit0/sourceHashes保持：正常2393/2400（99.71%）、扰动2398/2400（99.92%），24组全部≥95%、最差正常斜倒629017为197/200；不能称全例成功。当前板仍getup-mechanical-20261012-v2/SHA4b9c82e5…，不重复部署或训练。完整结果formal-result.json及RL formal-summary，下载归档SHAfec442bf5486be12e69be42e5fb2ba5057a79efb12fcca1bf8cfb73445203ded；逐帧4800轨迹远端1.7GiB保留，本轮仅下载完整分组结果/manifest/logs归档，不能说全部逐帧证据已下载。用户另确认坐姿站起会磕头，与此起身模型分开诊断；App0.9.43/code62取消坐下预备/站起及动态坐站按钮，只保留倒地起身作为启动入口，完成后摇杆行走。模拟器mock回归通过，真实手机新界面待验证；没有本轮代发实机动作或修改板控制。

> **2026-10-12 新限位起身模型已按用户要求提前部署（覆盖此前未部署状态）：release getup-mechanical-20261012-v2，模型SHA4b9c82e5bbd003a465b9bf861d4e7020014e7781d311f3780001e5257a588838。用户明确“先部署，再看验证进度”，正式仿真未结束，不称已通过正式/实机验收。仅替换起身ONNX/metadata和Rust二进制f41d2271…（增加新机械训练合同严格校验：14目标/物理范围、任务、执行revision；控制算法不改）。现行kissqy机械限位/EMA旧.45新.55/raw历史/最少3有效秒+直立稳定1秒/P5D20保持；其它模型/网页/helper/标定EEPROM/IMU/BLE身份/持久增益/15舵机profiles逐项保持，tofd/rsyslog仍masked。127Rust检查通过/1外部忽略，ARM构建通过；板纯推理观测误差0/输出2.980e-8rad，机械目标校验/8入口available/20只读卸力快照通过，未代发动作。首v1在切换前因上传二进制没有执行位拒绝，生产服务未变，v2显式chmod755后完成，旧getup4750-v2保留回滚。完整部署证据已下载哈希核验，附件/平台/2026-10-12/getup-mechanical/deployment.json及download-audit.json。**

> 正式仿真launcher718123仍在运行、无新训练或GPU；2026-10-12T05:55:43.328657+00:00快照正常1810/2400、扰动1859/2400，合3669/4800（76.4%）。两条件各完成seed517031六姿和seed629017前三姿（9/12组）；已完成每组≥95%，seed517031倒置正常199/200、扰动198/200，其余已完成组200/200。当前正在seed629017右侧倒地，尚余斜倒/倒置；不把未完成部分或本轮仿真称实机已成功。进度附件getup-formal-progress.json；不要重复启动718123。

> **2026-10-12 起身现行限位首轮完成/扩大仿真验收（最新覆盖）：** 双卡0/1、2048环境300轮已完成，job.exit0，候选SHA4b9c82e5bbd003a465b9bf861d4e7020014e7781d311f3780001e5257a588838；当前新物理/机械目标/编码合同下同初态seed15601基线0/12→候选12/12。fresh137031/249017正常47/48、扰动+一帧速度延迟48/48；唯一正常失败oblique249017/1，末倾63.55度稳定率0，不能称所有起身问题已修复。末3秒目标最大单帧正常.08789度/扰动.17578度（目标统计非实际响应），更新normalizer随正规导出，Torch/ONNX误差2.861e-6rad。完整round1归档SHAfef2443e…下载核验解压，226源hash逐项保持、screen initialQpos逐值相同、round1-final-audit.json。已独立启动CPU冻结正式仿真验证launcher718123，fresh seeds517031/629017，每姿每seed200例，正常/扰动合4800例，每组≥95%才通过；无GPU/新训练/硬件访问/部署，当前尚未有正式结果。所有限位/标定EEPROM/PD/BLE/其它技能及网页保持，仿真不是实机验收。证据平级RL artifacts/getup-mechanical-training-20261012/{round1,round1-final-audit.json}，服务器evidence/mechanical-formal。

> **2026-10-12 起身限位双卡微调已进入正式训练：** 最新独立launcher557967，0/1卡2048环境，日志已到59/300轮（此为阶段快照，不是完成）。f64v2双卡各64环境5轮smoke/正规导出/修正后64×40真实目标链通过，目标误差1.1667e-7rad/raw历史0；参数正常、Torch/ONNX含归一化。启动链证据归档SHAca8cf6a1…已完整下载核验解压；旧CLI与审计参考混精度失败日志保留，未重复基线或已完成f64 smoke。阶段结束自动六倒姿筛选及条件fresh回归，尚无合格候选/不部署。用户指定0/1驻留任务保留，无硬件访问、限位/标定EEPROM/PD/BLE/其它模型网页均保持。证据RL artifacts/getup-mechanical-training-20261012/startup-evidence.tar.gz及evidence/mechanical-training。

> **2026-10-12 起身重训启动链核验（最新覆盖）：** 按用户指定GPU0/1，独立case最新launcher557967。初始双卡CLI解析问题未进入训练；f32与f64 smoke均正常5轮/正规导出，后续动作链检查参考NumPy混精度导致假一步偏差，已定位纠正，不再归因为训练输出错一步。当前训练目标使用Rust同f64 EMA/HOME/裁剪/4096编码再转f32到BAM；修正参考后实际64×40目标最大误差1.1667e-7rad/raw历史0、4配置检查再通过。失败启动源码/日志分别保留attempt1-cli/attempt2-audit/attempt3-audit-reference；旧进程均退出，基线0/12及已完成f64 smoke复用不重跑。继续每卡1024×300第一轮，未完成、无合格结论；不动实机/不扩大限位/不部署，保留所有设备设置。用户此次新训练授权覆盖旧停止训练要求，GPU0/1驻留进程保留。证据平级RL artifacts/getup-mechanical-training-20261012。

> **2026-10-12 当前机械限位起身重训（用户新授权，已启动流程未完成）：** 用户明确按现行限位重新训练，并指定GPU0/1，覆盖此前停止训练。独立getup-mechanical-train-20261012 launcher496169；首482988双卡CLI解析失败、未开始训练，已保留源码/日志，改为仅可见0/1上的gpu-ids all后继续，已有基线导出/12例回放复用不重跑；配置4/4通过，getup4750 checkpoint SHA578ef3a2…保留actor/critic/normalizer初值，重置Adam和迭代，lr3e-4，原后期六倒姿覆盖/v3头部目标。仿真物理关节止挡和HOME+EMA绝对目标均对齐kissqy范围，raw历史/EMA.45/61维/14输出/4096步当前零点编码保持；CPU摩擦修订保持，body-floor接触不声称完整实物。GPU0/1用户授权保留驻留低利用率任务，不干扰他人，每阶段留12GiB余量；先每卡64×5smoke/正规导出Torch-ONNX及活跃动作链，再每卡1024×300第一轮微调，共2048环境/rank seed61217/61218；小样本六姿配对及条件fresh回归后再判断，不自动扩大或部署失败候选。未完成/无效果结论，不代发实机动作/改板权重、保护、标定EEPROM、增益或BLE。来源/进度见平级RL artifacts/getup-mechanical-training-20261012及远端evidence/mechanical-training。

> **2026-10-10 相邻帧/平滑覆盖CPU审计完成（离线，最新覆盖）：** 独立sit-temporal-coverage-audit-20261010 PID3956052已完成/job0/sourceHashes保持；无GPU/optimizer/训练/硬件访问。16条成功baseline轨迹构造6352对20ms相邻输入，严格同trial/同0～4、8～10、14～16秒窗口不跨断点；原real525/pairs517保持。raw反事实144组逐trial/关节p95/max已保存，stress3成功例首次2～3秒膝raw p95均值baseline15.740/13.126度，Critical15.650/13.079、control15.639/13.094；Critical平均输出漂移降低却相邻漂移p95左/右.291/.280度，高于control.197/.221，不可将均值拟合当抖动修复。CPU三actor同512对无放回seed73109，正规MLP含normalizer逐值保持、anchor和导出ORT parity通过、保存完整梯度向量197788参数：原real-temporal与关键raw膝top5%平方帧差proxy梯度cos约.090/.092/.091；关键sim绝对平滑cos约.741/.743/.742，但其基线梯度也非零且会压低正常成功动作，不能凭此直接加权训练/压平全部反馈。关键保留梯度cos control−.016/Critical−.039；teacher-delta梯度很小且cos−.048/−.090；仅本512对冻结输入局部方向非PPO更新/独立验收/实际闭环frame门，也非唯一故障原因。完整独立root证据SHAf7fdc7f8cc5c239567c0eb776ea1acc78caf275926803be430dd9091a15b14ba下载核验解压，全部manifest hash核验、temporal-coverage-final-audit.json；本地后续补充EMA.45+HOME+clip递推冻结反馈target统计，baseline与真实已捕获target最大容差3e-7rad通过，局部新增统计/源码hash另保存，重建NPZ四数组与远端原输入逐值一致（zip字节不假称同hash）。这是冻结baseline反馈的action递推，不是新动力学回放/实机PD。当前无活跃训练/合格候选，不重复3956052/CriticalReplay/旧case。下一先在同连续缓存和三actor做可微EMA+HOME+clip的真实target帧差proxy梯度，比较real-temporal/关键平滑/teacher-delta方向，raw proxy不能当实际target；必须保留跨帧图/正确初始filtered、基线回放target数值核对，控制内存并保持参数normalizer/源hash，无optimizer更新/新GPU。确认影响实际target且不压坏示范动作后才设计单项小预算训练，先测试64×5smoke正规导出normalizer，勿盲100轮重复Temporal100/ProcessV2等失败。所有最新平台技能/二进制/无EMA合同/独立增益与tofd和rsyslog masked保持，旧EMA候选不可覆盖当前kissqy入口，仿真非实机验收。

> **2026-10-10 CriticalReplay20完成且未合格（离线，最新覆盖）：** launcher3847932已结束/job.exit0，8检查、64×5smoke与1024×20正规导出通过，trainseed57121，Adam40/160、normalizer逐值保持/参数finite/std.099879，全部sourceHashes保持。完整独立root归档SHA227bab3ed59b4be251f1b0d77fb46068d49805d7738ca179bbd296bc420b72c6已下载核验解压，critical-replay-final-audit.json。四条件27/3/13/12、相对baseline无成功丢失，但stress多余行程增加2.366%、左膝p95帧跳28.788→30.999（+7.68%），过程/frame门失败；相对同20轮WeakProcess control丢5个stress成功例、行程增加10.70%，不部署、不加轮、不扩大holdout。8111/9001已经是拟合数据，这80例仅in-fit；恢复基线成功不能冒充独立验收或改善。新增CPU相同baseline观测反事实16例全帧对比：CriticalReplay/原control首2～3秒raw RMS.2941/.3890度，保留pool.2341/.2640度，全程.2153/.2162度；覆盖改动确实降低这些拟合状态上的输出漂移，却没有降低闭环多余行程和帧跳，不能以raw误差较小宣布姿态更好或唯一原因。诊断源audit-critical-replay-final.py首次整批ORT被固定batch1拒绝，改为逐帧后完成，无模型更新/仿真重跑/硬件访问，critical-replay-output-drift.json保留全16例统计。当前无活跃训练/合格候选，所有已完成旧case不重复。下一先用缓存baseline全帧核对各关节相邻输出差与漂移差、关键过渡尾部而非均值，审当前real-temporal与关键sim连续帧覆盖及其梯度是否指向实际frame门；只CPU冻结模型诊断不更新参数，不把反事实帧差等同闭环或实物响应。证据支持才设计单因素小预算修复，先测试64×5smoke正规导出normalizer，不盲100轮改系数/重复已失败CriticalReplay/FullBatch/CriticWarm/ProcessV2。CPU修订dof-id-friction-force-v1必须保持，旧物理数字不可混配。平台仍按最新observer记录保留kissqy-pick-getup-v1所有技能/二进制、坐站1c949无EMA/站起1秒回行走、新技能P5D20/行走持久P6D20、独立增益/标定EEPROM/BLE、tofd和rsyslog masked；旧EMA.45/clip-v3候选不可覆盖当前无EMA入口，诊断非当前板结论。仿真非实机验收，无主板访问/自动运动。

> **2026-10-10 闭环分叉证据与CriticalReplay20启动（离线，最新覆盖）：** SSH恢复后仅重传既有capture归档，SHA3dfe7a1b下载核验/解压、所有sourceHashes核对通过；CPU3766837共96例全qpos/result严格匹配原screen，未重跑训练/抓帧，full-batch-capture-final-audit.json。analyze-full-batch-failures.py已完成：两low失败在首次2秒起身后2.50/2.52秒出现连续5帧>1度倾角差；相同BASELINE观测含原raw-history反事实FullBatch输出此前RMS.2465/.2453度、最大1.187/1.176度（neck_pitch），control此前RMS.2141/.2096度、最大.931/.916度（left_hip_pitch）且最终成功。stress9001/3两candidate失败，分叉2.50/2.58秒，raw RMS.2916/.3028度、max2.235/2.682度，最大joint分别ankle/knee。说明权重更新在相同反馈前已有偏移，之后闭环放大；>1度仅诊断marker、成功control也会偏离，不能把marker当失败或head/leg当唯一首因，旧物理残差cap结论仍不可混配。失败控制器未部署。当前独立sit-critical-replay-control-20261010 launcher3847932已启动，GPU6启动19MiB、8测试通过；只改变offline保留示范分布，原3000 simulation保留，再从修正物理BASELINE成功的low13+stress3共16例6400帧pool（0～4/8～10/14～16秒）独立seed61007重采3000追加，均匀抽样期望50%原/50%关键过渡。real525帧/pair517逐值不变、seed.pt/teacher anchor/所有PPO奖励探索增益EMA/HOME限位/normalizer保持，WeakProcess4mini/24step/2epochs；trainseed57121，同前20轮control，先64×5smoke正规导出/normalizer/Adam40核验，再1024×20/Adam160，不重新训旧control或FullBatch。输入状态涵盖新失败附近，属于保留覆盖假设非已证明改善；关键数据源8111/9001从现在明确属于拟合数据，其80例筛选只in-fit，必须未用新seed再正式≥2seed每组200环境95%，不能当独立验收。所有源/参考回放/dataset hash锁定，dataset-audit与launch已保存，跟踪evidence/critical-replay/{progress,job.exit,comparison,job.log}不可重复启动；尚无训练效果/候选部署结论。不盲100轮改系数或压平全部反馈。保留最新kissqy-pick-getup-v1所有技能/二进制、坐站1c949无EMA/站起1秒回行走、新技能P5D20/行走持久P6D20、独立增益/标定EEPROM/BLE、tofd/rsyslog masked，旧EMA候选不可覆盖当前入口、非当前板结论。无主板访问/实机运动，仿真非实机验收。

> **2026-10-10 FullBatch抓帧传输状态（最新覆盖）：** CPU抓帧3766837已完成/job.exit0，未重跑训练；远端capture归档SHA3dfe7a1b1359846d12c143dfec319057c7c2a6e9dd48d38110589bbb2a29d0ac已生成。SSH传输再次超时，精确停止本次scp29208并保留部分文件，尚未完成本地SHA/解压，不称完整capture证据已下载。已核验FullBatch20完整证据SHA0c717c2a保持、两组失败不部署。恢复连接后只重传归档，运行已准备analyze-full-batch-failures.py做同观测反事实/分叉定位；不重训或重跑96抓帧，不访问实机。

> **2026-10-10 FullBatch20对照失败与闭环抓帧启动（离线，最新覆盖）：** sit-full-batch-control-20261010 launcher3687104已完成/job.exit0/sourceHashes保持；6检查、两分支64×5smoke正规导出/normalizer通过，seed57121/1024×20。Adam张量实际control160次/FullBatch40次（smoke40/10），std约.09993/.09996、normalizer逐值保持、参数finite；变更仅num_mini_batches4→1但aux hook与Adam次数耦合不能称纯降噪。基线27/3/13/12，control27/7/13/12却丢stress sit_cycle9001/3、膝多余行程少7.53%、左膝p95帧跳28.788→32.158（+11.71%）失败。FullBatch27/3/11/12，亦丢stress sit_cycle9001/3及low sit_cycle8111/0、9001/0，行程少5.32%但左膝p95→32.629（+13.35%）失败；相对control又丢stress4+low2、行程反增2.39%，均不扩大/部署/加轮。不能因stress成功总数相同或超过5%过程门忽略逐例退化/frame门，也不能用此否定全部大batch方法或证明唯一奖励根因。完整源/模型/metadata/四条件240回放及训练checkpoint/log/contract SHA0c717c2a下载核验解压并逐项sourceHash核对，full-batch-final-audit.json；上轮启动后SSH一次超时仅连接波动，任务已完整完成，未重复。当前无GPU训练/合格候选。下一CPU完整反馈抓帧已启动evidence/full-batch-failure-capture launcher3766837：baseline/control/FullBatch×low/stress共96例，保持原scenario顺序/seed/trials，新增capture-policy-frames，完整qpos/result逐值匹配原screen、初态不重排，记录raw/filtered/target/obs定位失败开始；无新独立验收/训练/GPU/板访问。已有launch跟踪不可重跑；待匹配证据后在相同baseline观测做两candidate反事实输出，区别初始输出漂移与反馈发散，不凭末段关节差宣称首因，不盲加系数或100轮。保留最新kissqy-pick-getup-v1所有新技能、无EMA/P5D20（行走持久P6D20）、坐站1c949/站起1秒交回行走、二进制/独立增益/标定EEPROM/BLE及tofd/rsyslog masked；旧ec665属历史不换回。旧EMA clip-v3诊断不能代表当前板模型或覆盖新入口合同。仿真非实机验收。

> **2026-10-10 FullBatch20配对对照已启动（离线）：** 独立sit-full-batch-control-20261010、launcher3687104，GPU6启动19MiB/新训练seed57121，共同seed.pt空Adam；6项RL算法唯一差异/训练及play完整配置/GAE/浮动基座DOF摩擦检查通过。FullBatch仅WeakProcess num_mini_batches4→1，1024×24步/2epochs/奖励actor-critic探索增益EMA HOME限位归一化合同保持；Adam与aux hook每轮8→2，每步6144→24576，耦合效应不称纯噪声隔离。两分支分别先64world5轮正规smoke导出/normalizer审计，再1024world20轮，按Adam step张量核对control160次/FullBatch40次（smoke40/10），检查finite与逐值normalizer。种子正规导出baseline；候选及control四条件80例各两seed8111/9001，同初态逐例匹配新DOF摩擦revision，评估相对未训练seed及20轮control的成功/膝过程/frame门；相同训练seed并不保证训练轨迹同初态，更新/随机采样消费会分叉。当前运行尚无效果/合格结论，勿重启或扩大100轮。首环境repr检查只因新闭包地址不同失败，改为仅规范地址保留其它配置后通过；首launch因GPU审计源未含CPU修正文件在hash清单前拒绝，未创建launch/PID/训练，随后补入当前CPU evaluator/BAM修正与finalizer并6测试通过才正式启动。完整sourceHashes包含训练源/seed/CPU物理/evaluator/导出/tests，跟踪evidence/full-batch-control/{launch,progress,comparison,job.exit,job.log}。本轮无主板访问/运动/部署；保留最新kissqy-pick-getup-v1所有技能与无EMA合同、行走持久P6/D20/新技能P5D20、独立增益/标定EEPROM/BLE、tofd和rsyslog masked。旧EMA clip-v3候选不可覆盖新入口。仿真非实机验收。

> **2026-10-10 正式规模24步梯度及价值项拆分完成（离线，最新覆盖）：** sit-formal24-gradients-20261010 GPU6启动19MiB、2GAE检查，PID3539382已完成/job.exit0/源hash保持。Main0/Main99×9017/12031各1024world连续21×24步，在0/192/480步取initial/first_edge/late，共12窗口×4随机6144样本批；reward误差1.19e-7/adv重构1.24e-5/分项gradient sum2.59e-6、ratio最大偏差5.44e-5，actor含normalizer逐值保持无optimizer step。mean网络critic投影初.613～.867、切换.676～1.291、后期.938～1.050，knee−.006925～.003161；正式规模下仍非只64world偶然，但投影非奖励占比或错误证明。随后CPU缓存价值拆分PID3610451完成/2新测试/source60NPZ保持/actor保持：直接current-state baseline=-V(s)，future remainder=critic GAE−baseline（含所有未来值/截止bootstrap/timeout/reset），reward GAE，共同总std中心化；48批总gradient与GPU相对差<=6.38e-6、分项sum<=1.73e-6。baseline投影初−1.089～.635/edge.024～1.160/late−.078～1.171，future同样相消，不能将合并critic全当错误状态baseline或删除。四批mean网络方向平均cos：初.052～.147/edge−.032～.070/late−.080～.046；完整批均值范数/单批RMS .426～.599，表明本批随机minibatch方向一致性弱，时间相关/同rollout分块不等于独立噪声辨识。两完整证据SHA609bfdd0/836e10fa下载核验解压，formal24-final-audit.json、formal24-batch-directions.json与全部checkpoint/源/12rollout/48分项vector/valueparts保持。所有诊断仅独立冻结ratio≈1首更新点，非历史1024训练minibatch或第二epoch；critic normalizer采样时正常更新，梯度期间固定。当前无活跃训练或合格部署候选，勿重复上述PID。下一依据低批方向一致性做小预算FullBatch对照：只将WeakProcess PPO num_mini_batches4→1，1024×24/2epochs保持，其余actor/critic/奖励/探索/gain/EMA/限位/HOME/normalizer不变；要明确每轮Adam/aux hook次数8→2、每步6144→24576是耦合效应，不能称只降噪纯实验。先配置差异检查/64×5smoke正规导出及normalizer审计，再新seed同预算20轮control与FullBatch配对四条件80例，相对未训练seed及20轮control逐例/过程/frame门，不预设有效/直接100轮；检查已有目录/PID防重复，GPU6<200MiB才能新阶段。不重复强过程/CriticWarm旧失败。平台最新已由其它用户授权任务部署20261010-kissqy-pick-getup-v1：新getup stand教师a481…/pick2b69…无EMA/P5D20，行走用户持久P6/D20、坐站1c949无EMA/站起1秒交回新行走保持。旧ec665仿真验收现是历史模型结论，不能说当前板仍ec665，不自动换回；保留最新二进制/技能/独立增益/标定EEPROM/BLE、tofd和rsyslog masked；本轮未访问主板/运动/部署，仿真非实机验收。旧EMA clip-v3训练候选不能直接覆盖新入口合同。

> **2026-10-10 正式规模冻结24步梯度核对已启动（离线）：** 独立sit-formal24-gradients-20261010，launcher3539382、GPU6启动19MiB，2项共同归一化/超时GAE测试通过。1024world沿用正式num_steps_per_env24/4随机minibatch每批6144，不更新actor/optimizer、不训练或导出候选；Main0/Main99两seed9017/12031，连续21个24步块，在initial/first_edge/late保存完整观测奖励值与逐batch分项梯度，核对reward/GAE/sum及actor含normalizer逐值保持，分开std与mean网络。critic normalizer在rollout按正常流程更新、每次梯度审计固定。范围仅冻结ratio≈1处独立rollout，非历史训练minibatch或第二epoch更新。已启动不可重复，跟踪evidence/formal24/{launch.json,job.log,job.exit,summary.json}；未完成无结果/根因/改善结论，全部旧失败不重复。保留最新20261010-kissqy-sitstand-restore-v1（1c949权重、无EMA/P5D20/站起1秒回新行走），保留二进制行走独立增益/起身ec665/标定EEPROM/BLE、tofd和rsyslog masked，不访问主板/发实机动作/部署。仿真非实机验收。

> **2026-10-10 缓存24步参数梯度与std/均值网络拆分完成（离线，最新覆盖）：** sit-cached24-gradients-v2-20261010 PID3480613完成/job.exit0/全部源hash保持，Main0/Main99×9017/12031，各起始、首指令切换、后期24步×64worlds，共12窗口；lambda.95/1共24个重构总梯度，分项sum相对误差最大5.92e-07，actor含normalizer逐值保持、无optimizer step/新训练/GPU/硬件访问。实际named_parameters核对197788参数，distribution.std_param前14；其总梯度平方范数占0.55～3.35%，分开后均值输出网络critic有符号总方向投影：起始.354～.728、切换.803～1.186、后期.920～.985，膝过程约−.01439～.000734。投影可负或>1，不是占比/奖励大小或唯一故障证明，价值baseline项即使期望无偏也可能有限样本方向显著。lambda1与.95总方向cos .939～.986；起始critic投影下降但切换大多升高，不支持直接宣称lambda1解决问题或据此删critic。此为64world缓存匹配上下文，不是正式1024world/24步/4minibatch（每批6144样本）历史更新；前轮512窗及保留项归因同样不能冒充原训练。完整源/checkpoints/输入/12向量/参数拆分及初失败源日志SHA2b1fe6af下载核验解压，cached24-final-audit.json。首PID3462729 launcher误引旧GPU脚本，被CUDA_VISIBLE_DEVICES空guard计算前拒绝，无GPU工作；独立v2修正完成勿重跑。当前无训练/合格候选/部署。下一核对正式样本规模的有限批方向稳定性：先审现有数据，必要时GPU6<200MiB独立1024world冻结rollout按24步分块、匹配全块共同归一化/4×6144mini-batch，取起始/切换/后期，核对reward/GAE/分项sum与actor含normalizer不变；只rollout不优化，明确非历史原minibatch。先排除64world有限样本归因再决定单因素小预算训练，不盲改lambda/系数100轮，不重复CriticWarm/ProcessV2失败。保留kissqy-sitstand-v1 SHA1c949无EMA/P5D20/站起1秒回新行走及独立增益/起身ec665/标定EEPROM/BLE；旧EMA诊断非当前板结论，不覆盖新入口合同。仿真非实机验收。

> **2026-10-10 奖励GAE/梯度分解完成及窗口纠正（离线）：** 独立sit-reward-gradients-20261010，2项超时重置/共同归一化测试通过，GPU6启动19MiB，PID3400968完成/job.exit0/源hash保持。Main0/Main99两seed9017/12031各64×512步：weighted奖励sum误差8.94e-8、normalized advantage重构误差<=7.01e-6、各项PPO raw梯度sum相对误差<=2.35e-6；actor含normalizer逐值不变，无optimizer step/训练。512步窗口中critic_bootstrap_value对总参数梯度方向的平均有符号投影.636～.746，knee_process .00011～.00238、目标加速度约−.000044～.000065；不是奖励占比/唯一根因，不据此删critic或改系数。重要适用边界：为覆盖完整循环本诊断用512步，但正式训练num_steps_per_env=24！前轮512梯度亦是独立长上下文，不代表原训练批。缓存同轨迹切成21个24步窗后critic对优势方差平均投影.022～.218（长窗.610～.715），属于优势方差非参数梯度；说明窗口选择显著影响归因。gamma.99/lambda.95名义指数衰减时间.336s仅启发，不能凭此宣布长credit更好。完整证据SHAe95c7bcf下载核验，reward-gradients-final-audit.json/完整rollout+first-batch梯度向量已保存；cached-context-audit亦保存。当前无运行训练/合格候选，不重复3400968或先前已完成诊断。下一优先使用缓存轨迹和原checkpoint在CPU重算真实24步窗口参数梯度（起始/指令切换/后期代表窗，共同std、sum对照），不必重复GPU采集；再决定信用时间等小预算单项控制。保留当前kissqy无EMA新平台、起身ec665及全部设备设置；无实机访问/运动/部署。

> **2026-10-10 冻结真实rollout梯度诊断完成（离线）：** 独立gradient-v2/v3，3项梯度裁剪/方向/继承Adam缓存与不改原参数检查通过，GPU6启动19MiB。起始actor及同一WeakProcess正式run的Main0/Main99各两seed9017/12031、64环境×512步，每run128指令切换，32个正式checkpoint minibatch与16个起始minibatch；只取真实rollout/returns梯度，未训练/导出候选。起始保留梯度为0（actor与anchor相同，非代码失效），PPO原始范数4.35～5.46→裁剪1，时间平滑0.006319。Main0离线保留0.0321～0.0337/在策略0.00599～0.00629/时间0.006308；Main99分别0.1169～0.1240/0.04148～0.05347/0.005716，PPO裁后仍1。复制继承Adam一step的PPO-only/合成方向cos Main0 .999875～.999881、Main99 .998677～.998800；辅助项实际改动更新有限，但范数/方向仅本批证据，非唯一故障根因/闭环验收。actor含normalizer、critic梯度审计期间和optimizer state逐值保持；critic normalizer按实际流程在rollout采集时更新，不能把它称全程冻结。初次PID3231406因PyTorch非叶缓存deepcopy报错，源/日志保留，修复缓存脱图后v2 PID3247083/v3 PID3268921完成job.exit0/源hash保持，全部勿重跑。完整初始含失败证据SHA2d6406a4及正式checkpoints SHA6e46f0a1下载核验解压，gradient-contributions-final-audit.json清单。下一拆分各奖励的GAE/PPO梯度贡献并校验和等于原总梯度，再决定有据小预算优化，不盲调系数100轮。旧EMA.45合同诊断，非当前kissqy无EMA板模型结论；当前无训练/合格部署候选，保留新平台/起身ec665/增益/标定EEPROM/BLE，无实机访问运动。

> **2026-10-10 修正摩擦后11监督候选及独立seed复测完成（离线）：** 独立sit-supervised-corrected-screen-20261010，14检查/362源模型基线hash核验通过，11现成导出均未重新拟合/训练；四条件每actor80例严格初态配对，修正基线27/3/13/12。仅Knee2初筛无丢成功/过程少8.00%、Gate3无丢成功/少6.42%且frame门通过；新seed33013/44017基线29/7/14/15，两候选同为29/7/14/15却都丢stress的sit_cycle/33013/2，Knee2过程反增5.29%、Gate3仅少1.82%，两者失败，不扩大正式验收或部署。其它9候选逐例退化或改善不足，全部结束勿重跑PID3144296/3180284/3189519；当前无活跃训练/合格候选。完整screen SHA93c21fa9…及holdout SHAd7e75494…下载核验解压，审计supervised-corrected-final-audit.json。结论仅属旧EMA.45/clip-v3候选在修正CPU物理的回归，非当前kissqy无EMA入口实机结论。保留最新kissqy-sitstand-v1、新行走二进制/独立增益、起身ec665及标定EEPROM/BLE，未访问实机/发运动或改设备设置。下一先量化PPO/离线保留/在策略保留/时间平滑各项实际梯度范数和方向，再有据小预算训练；不要继续盲调100轮。仿真非实机验收。

> **2026-10-10 修正CPU物理后监督候选复核已启动（离线）：** 独立sit-supervised-corrected-screen-20261010，launcher3144296，14项CPU摩擦/回放检查通过；11个现成导出（Broad1/3/10、Knee1/2/3/10/30、Gate1/2/3）原文件SHA及61维/HOME/EMA/接触/限位/执行器合同匹配，不重新拟合或训练。两CPU worker，修订dof-id-friction-force-v1，复用相同源/模型的修正后基线27/3/13/12，四条件各候选80例严格初态配对；362项源/模型/基线报告哈希启动前及结束后核验。待全部结果，尚无合格结论/部署，旧物理失败不能替代此次结果。保留最新kissqy-sitstand-v1无EMA入口、新行走/独立增益及起身ec665，本批旧EMA候选不能直接覆盖其合同。未访问实机、不发动作/改标定EEPROM或BLE；仿真非实机验收。已有launch不能重复启动。

> **2026-10-10 CPU摩擦编号修正与重测（最新覆盖）：** 上游CPU BAM把DOF摩擦efc_id匹配关节id；浮动基座joint1～14/dof6～19导致串位，GPU用dof正确。新增scripts/bam_cpu_controller.py仅修正积分前摩擦预算，保留上游控制/电压/目标限速；CPU skill接入、summary修订dof-id-friction-force-v1、缓存拒绝旧revision。14本地/14服务器检查通过。GPU6空闲19MiB时4world实际装配595命名字段无差异；480冻结负载样本摩擦最大差0.259455→3.19e-5Nm，电机力矩差7.70e-8Nm。非完整CPU/Warp动力学或实机PD一致证明；旧CPU成功率及0.25°闭环漂移诊断属旧物理历史，不能与修正后混配。修正后重测12个现成PPO、不新训练：基线正常27/32、stress3/16、low13/16、high12/16。LowExplore初筛保全部成功且过程少5.74%，新seed12017/24029为28/8/15/15，丢stress1旧成功、过程仅少1.04%；Upright初筛27/6/13/12，无丢例/少5.74%，新seed28/8/15/15无丢例但过程仅少1.36%，未达5%门。其它逐例退化或frame门未过，全部结束不部署/不扩大正式验收。
>
> **起身新物理检查：** ec665正常/受扰/组合接触无扰各48/48、组合接触受扰47/48、原/v3坐姿各16/16；组合失败right_side/9001/0末段43.55°、目标逐帧最大0.0117°，头壳接地/左脚支撑，倾斜停住而非反复抖。contact-repair同时加入自碰撞与XML目标裁剪，非板上getup合同。严格复现六姿seed9001各4trial隔离共96例：原/仅自碰撞/仅裁剪各24/24、组合23/24，失败初态与原记录逐值匹配；不能怪某项单独有错，不据此改板权重。首right_side-only16例改变RNG前序未复现原初态，保留归档不当反证。七份完整证据已下载SHA核验，cpu-friction-final-audit.json清单；截图/接触对/逐帧隔离记录亦保存。
>
> **平台最新状态与下一步：** 其它用户授权任务已部署20261010-kissqy-sitstand-v1（SHA1c949…、无EMA、P5/D20、站起1有效秒回行走），保留该坐站/行走二进制及增益；本任务旧EMA0.45/clip-v3候选不能直接塞进此入口合同。全部标定/EEPROM/BLE身份/其它设置保持，本轮未实机运动/新部署，仿真非实机验收。所有PID3002200/3023503/3032392/3045828/3070559/3080998/3094345已结束，无运行训练/合格候选。下一先用修正物理复核尚未重测的现成监督/膝专项/门控候选，不重复拟合/PPO；仍无合格再做PPO/保留/时间平滑梯度贡献诊断并小预算优化，勿盲改系数100轮。

> **2026-10-10 分组漂移诊断最终（离线）：** group-drift-probe64例完成/job.exit=0、source-check通过，完整证据SHAfe69f74f…下载核验。仅腿部输出差±0.25°：stress8/16、low6/16，分别丢1/6旧成功；仅头颈四关节输出差±0.25°：stress8/16、low6/16，亦丢1/6且低gain丢例集合不同。两组完整反馈动态计算、未固定关节姿态，因此说明本批腿及头颈微小更新均可能破坏低gain过渡，不能归因唯一膝或头颈。head-only的stress膝行程下降约4.25%却成功退化，仍失败、不部署。两组同时有小修正时±.25°low9/16比单组6/16高，不能用各关节独立优化来假设整体必改善。所有probe均为双actor因果诊断非可部署单模型；已完成勿重跑launcher2953811/2937195。当前无新GPU训练/未部署/未发实机动作。下一步先核查尚未证明的CPU/GPU实际动力学一致性：相同低gain初态、qvel、EMA/raw历史与动作时序，对比BAM力矩/运动/接触配置并定位首偏差；现有64×80仅证明动作增益算术合同，不能冒充动力学等价。GPU6仅空闲<200MiB才做小批诊断，不干扰别人。不要再凭奖励系数或平均保留loss重复100轮。保留kissqy新行走/P5D20、起身ec665及其它设备设置；仿真不等于实机验收。

> **2026-10-10 残差上界因果对照完成（离线）：** CPU两档四条件合计160例/job.exit=0、两policy与基线初态严格配对，±0.25°为正常28/32、扰动9/16、低gain9/16、高gain13/16；扰动仍丢1、低gain丢3旧成功例。±0.5°为28/10/3/13，扰动总数增加却丢1旧成功、低gain丢9。两档均不合格/不部署，且为双actor诊断非单模型导出，accepted=false/policySha=null/composite含两hash及cap。完整证据SHA8ea5a285…下载核验、原driver/模型源hash保持；初次wrapper失败SHA85b9e6e1…亦保留下载。±0.25°的三个低gain退化例均为坐姿起始首次站起，基线窗口倾角3.65～3.80°→probe35.20～35.57°，3.50～3.60秒开始持续偏离；当时接触总数两边同为2，不能从接触总数断言同一接触形态。输出漂移虽小仍可改变低gain过渡闭环，不能说限制平均漂移即可解决，更不能断言neck/knee唯一根因；观察到的关节轨迹差是发散结果非首个因果输出。当前分组探测CPU-only launcher2953811、remote sit-boundary-gain-control-20261010/evidence/group-drift-probe：保留完整反馈及原actor，仅分别在legs或head四关节组保留final输出差±0.25°，另一组仍由原actor动态计算；两组×stress/低gain×16共64例且保存完整policy帧，用于隔离哪组漂移触发失败。不新训练/占GPU/硬件运动/更改服务或参数；下一步待因果结果决定训练约束。保留kissqy新行走、P5/D20和其它设置，仿真非实物验收。

> **残差诊断启动状态纠正：** 初次CPU wrapper在进入仿真前缺get_outputs报错，旧PID2930671/job.exit=1/源码日志已保留residual-cap-wrapper-failed.tar.gz；已补齐原ORT输出接口，在独立evidence/residual-cap-probe-v2启动PID2937195，两档±0.25/0.5°四条件配对诊断，无新训练/GPU/硬件操作。勿跟踪旧PID或把该脚本错误当模型失败。

> **2026-10-10 17:45 BoundaryGain最终失败及早期漂移诊断（离线）：** 100轮完成/job.exit=0，正常28/32、扰动8/16、低增益2/16、高增益13/16；扰动丢2、低增益丢10旧成功例，扰动膝多余行程平均增加10.1%，不部署。64×80实际链确认kp .8～1.2/std .1427且compute不重置，EMA/HOME/clip误差1.19e-7、raw历史0；normalizer逐值保持、actor/critic finite、执行合同及训练源哈希保持。完整证据SHA d867ef9d…下载核验。原有64环境5轮smoke补测低增益10/16（基线12），丢sit_cycle/9001/1和3，证据SHA575d0589…下载核验；此smoke不是1024环境主训练的中间checkpoint，不能称同一学习曲线。冻结基线成功低增益观测上final输出RMS仅0.280°、左膝99分位差1.840°；拟合simulation RMS0.246°，平均保留损失可能掩盖局部敏感性，尚非唯一根因。逐例stand_cycle/9001/1第一坐窗口正常，下一站窗口倾角基线3.70°→35.45°，说明双向过渡失效而非初态即倒。已启动CPU残差上界因果探测launcher2930671：原actor与final读取相同完整反馈，仅将新旧输出差限制±0.25/0.5°，两档四条件各80例配对；不新训练/占GPU/碰硬件，不是单模型导出或可部署候选，不能按结果直接换服务。结果后再决定训练trust-region等改法，勿继续盲改100轮系数。板上新kissqy行走/二进制及17:36用户P5/D20保持；无实机动作/设置更改。仿真不是实机验收。

> **2026-10-10 BoundaryGain分桶对照已启动（离线）：** 7项检查通过、GPU6启动前19MiB/0%，独立launcher2822519，服务器sit-boundary-gain-control-20261010。先64×5smoke/正规导出/normalizer以及64×80步实际增益与EMA/raw历史审计，通过才1024×100及四条件80例配对筛选。kp五桶.8/.82/1/1.2/uniform各20%，其它WeakProcess奖励/探索std/critic初始化/物理与61维接口保持，尚未完成、不部署；分桶有意改变gain分布/均值，只验证这种覆盖修复。已有失败权重保留，未发实机动作/改标定EEPROM增益/BLE，新行走release继续保留。

> **2026-10-10 冻结增益扫描完成（离线）：** 64例完整、job.exit=0、每gain两policy初态逐值配对。seed9001站起始循环各4例：基线8档全部4/4；ArrivalGate在0.8为0/4（末站34～35°）、0.82为2/4（两例约34°）、0.85～1.2各4/4（约3～4°）。显示本批微调后低增益裕量缩小；单seed非全面可靠性/实机P/D证据，跨gain初始静置亦随gain变化。完整证据SHA0a9d6a17…下载核验，逐帧JSON/NPZ完整；旧gain-curve driver小数目录with_suffix导致同model日志共用0.log/1.log（未丢结果/轨迹），已仅修正本地后续日志命名，原source/driver哈希归档保留，不重跑。下一BoundaryGain只改变WeakProcess训练kp采样：.8/.82/1/1.2/连续uniform各20%，刻意偏重低边界、平均kp略下降，故不是单纯等均值稀有度实验；奖励/探索/critic/全部动作接口与物理保持，先测试/64×5smoke+活动链核对，再1024×100严格四条件。未部署或发实机动作，保留当前新行走及增益配置。

> **2026-10-10 训练链审计通过/冻结增益扫描已启动（离线）：** GPU64环境80步WeakProcess实际kp_scale0.8094～1.1893、std0.099、BAM baseKp5、kd_scale0.9043～1.0994；compute间增益变化0，有效kp逐值等于base×scale。目标EMA+HOME+clip最大误差1.19e-7rad、raw prev_action误差0，未发现随机化被重置/动作链不一致。本批单seed诊断非实机P/D辨识，也非完整CPU/GPU动力学等价。冻结基线与失败ArrivalGate启动CPU增益扫描launcher2769666（原服务器sit-arrival-gate-control-20261010/evidence/gain-curve）：0.8/.82/.85/.9/.95/1/1.1/1.2，seed9001站起始重复坐站各4例，每增益两policy同初态，共64例；跨增益初始静置本身受gain影响，需要区分。此阶段不训练/不占GPU/不部署，曲线出来再决定边界采样等因果对照；不重复已失败PPO。实物/设置保持、后续保留新行走release。

> **2026-10-10 ArrivalGate100最终失败（离线）：** job.exit=0、完整配对正常28/32、扰动9/16、低增益0/16、高增益13/16；扰动丢2旧成功例，不部署。低增益stand_cycle/9001/0末段仍34.38°，不能将几何奖励门控的静态改善冒充闭环修复。normalizer逐值保持、finite/源核验通过；完整证据SHA45a13ebe…下载核验。下一步暂不继续改奖励，核查实际GPU训练中的增益随机化是否保持及EMA/HOME/clip/raw历史链路；64环境80步只仿真，无实机运动/设置更改，保留当前新行走release。奖励局部解/critic/探索各项未独立解决，不能预设唯一原因。

> **2026-10-10 ArrivalGate奖励对照已启动（离线）：** 11项检查通过，GPU6空闲，launcher2688946、sit-arrival-gate-control-20261010；先64×5smoke/正规导出/normalizer核对，再1024×100和四条件80例严格同初态全过程筛选。只门控参考ramp完成后的腿姿、两高度Gaussian及stillness正奖励（12～20°平滑降到0），不改actor观测/EMA/HOME/限位/执行器，也不抑制过渡前倾或删除upright/L1导向。根据静态奖励审计提出局部解假设，未证明唯一根因、未完成/不部署；CriticWarm最终失败证据已保存。无实机动作/参数改变，保留其它新行走release及独立增益。

> **2026-10-10 17:00 CriticWarm最终失败及奖励静态审计（离线）：** 20轮critic-only+100轮PPO及全部配对筛选结束、job.exit=0，正常28/32、扰动9/16、低增益0/16、高增益13/16；扰动成功总数保持却丢2个旧成功例，仍失败，不部署。critic-only actor逐值审计通过、最终normalizer保持/critic finite/源哈希核验通过，完整证据SHA c6421c2f…下载。低增益stand_cycle/9001/0末段倾角34.77°。9项几何任务奖励静态审计（速度置0、alpha=端点指令、非总回报/PPO advantage）正确理想坐站各14.25、错指令各1.09；同trial基线3.62°得13.29，CriticWarm34.77°仍得8.78，其中stillness1.77、composite仅0.00192。错误姿态未超过正确姿态，不能说奖励全局最优错误；但仍有部分到位分可拿，与20°验收不一致，可形成倾斜局部解。下一项ArrivalGate只把端点四项正奖励（腿姿/两高度Gaussian/stillness）在12～20°平滑衰减到0，ramp未完成不门控，保留upright/bootstrap/L1梯度、actor全部接口与物理。不是实机角度保护；待测试smoke/100轮严格回归。此轮未发实机动作/部署/更改标定增益EEPROM/BLE；当前平台新kissqy-walk-v1需保留。

> **2026-10-10 价值网络预校准对照已启动（离线）：** CriticWarm9项测试通过、GPU6空闲，独立launcher2586030，服务器sit-critic-warm-control-20261010。先64×5冒烟确认actor全部state逐值未变/导出；正式1024环境20轮只训练critic（Adam1e-3、当前奖励returns，主optimizer/actor辅助hook不运行），然后100轮同WeakProcess PPO，合计120轮。旧actor/normalizer/探索std0.1/奖励/动作及物理不改，区别是先校准critic，不能与更长actor训练混称。校准计数与optimizer随checkpoint保存，每次critic-only更新逐值审计actor。尚未完成/不部署；这验证一种价值适配处理，不预先宣布根因。LowExplore完整导出元数据合同/normalizer本地复核通过但闭环失败，保留。板上新kissqy-walk-v1及各自增益设置以observer最新记录为准，后续部署不能覆盖；此轮无实机动作/参数改变。

> **2026-10-10 16:45 LowExplore100最终失败（离线）：** 64×5冒烟后1024×100训练及全部配对CPU筛选结束，job.exit=0；正常28/32、扰动8/16、低增益8/16、高增益13/16。相对WeakProcess低增益0/16有改善，仍丢基线12/16中的4个成功例；扰动丢2例并新获1例，总数8不能掩盖退化。扰动两膝多余行程48.097/40.904→45.125/38.677°，平均降约5.84%，仍不合格、不部署。最终normalizer逐值保持、finite、探索std均值0.0297568rad、源哈希保持；完整证据下载SHA07508ced…核验。下一项拟冻结actor只校准critic后再同预算PPO，验证旧critic与监督修复actor/新奖励适配；未证明唯一根因。不发实机动作/改标定EEPROM增益身份；读取observer最新记录发现其它已授权更新已将release换为kissqy-walk-v1，后续任何坐站部署需重新核对并保留该新行走实现，不回退旧speed-040。仿真不是实机验收。

> **2026-10-10 低探索因果对照已启动（离线）：** LowExplore 7项配置/方差测试通过，GPU6启动前空闲，独立launcher2516396，服务器sit-low-explore-control-20261010。仅对WeakProcess探索std改0.02～0.03rad（load后首次act前及update后夹紧），旧actor均值/critic初始化、奖励、61维归一化/动作EMA/HOME/限位/自碰撞/增益采样保持；先64环境5轮smoke，正规导出校验后1024环境100轮及四条件80例配对全过程筛选。尚未完成最终筛选，不部署。Upright与Capped最终normalizer逐值保持、finite、完整元数据合同无差异（upright-final-contract-audit.json）；失败不能归因导出normalizer改变。已下载两组完整证据并SHA核验。此次无实机动作/部署/设备设置改变，持续检查跟踪现有PID、不重复启动。

> **2026-10-10 16:30 Upright两组对照最终失败（离线）：** 两组均64环境5轮冒烟/正规导出后1024环境100轮结束，job.exit=0。Upright正常28/32、扰动8/16、低增益5/16、高增益13/16；封顶UprightCapped为28/32、10/16、3/16、13/16。两组均丢基线成功例，不部署；后者成功总数提高仍不合格。相同低增益stand_cycle/9001/0末段倾角34.92/34.31°，未消除倾斜稳定解。封顶只将cost上限设4（weight−2），value loss31.90→1.28、reward−181.49→131.09，闭环仍退化，不能以损失/奖励好转当动作修好。normalizer保持、参数finite，封顶源哈希核验通过；完整证据下载SHA校验，位于平级RL artifacts/process-repair-v2-20261010/{upright-final,upright-capped-final,upright-diagnostic-audit.json}。下一小规模LowExplore仅对WeakProcess降低探索std至0.02～0.03rad，仍旧critic/奖励/全部物理与部署合同，准备后先测试冒烟，再100轮；价值模型与修复actor适配亦未排除。未发实机动作/部署/改标定、EEPROM、增益或BLE；板上仍speed-040-v1，仿真不是实机验收。

> **2026-10-10 16:00 接续优化已启动（离线）：** GPU6已恢复空闲19MiB/0%，Upright端点倾角奖励对照launcher2353603启动；64环境5轮smoke正规导出通过、参数finite且normalizer逐值保持，随后1024环境最多100轮正在运行，四条件配对全过程评估完成后才判断效果，不部署未完成或退化候选。只在参考ramp完成后增加12°以上倾角成本，保持过渡动作/61维/EMA/HOME/限位/物理合同；未发实机动作或改标定增益。板上起身已只读核对同一ec665权重，最新release speed-040-v1，不需重复替换。证据artifacts/process-repair-v2-20261010/upright-control/；每15分钟接续检查保持。

> **2026-10-10 过程优化与分项诊断最终结果（最新覆盖）：** SitStandProcess-v2的1000轮及CPU/GPU评估全部结束，job.exit=0，训练26分42秒。严格服务器同初态名义28/32→26/32、扰动9/16→6/16、低增益12/16→0/16、高增益13/16→13/16；扰动膝多余行程虽少15.2%，成功退化，selected=null，不部署。最终normalizer逐值保持、训练源哈希未改；两个最终ONNX文件哈希不同但全部initializer及3000条输出完全相等。四个各100轮对照Coverage/WeakProcess/Temporal/Sensor都丢成功例，低增益0/0/0/4（基线12/16），不能唯一归因强平滑税；另状态腿姿奖励参考StateReference100轮也为28/32、8/16、0/16、13/16，排除候选，但未证明或排除计时隐状态的全部影响。仅增动作EMA亦退化，不改服务滤波。原起身ec665…六倒姿正常/物理扰动各48/48，自碰撞诊断亦各48/48，原坐姿/当前v3坐姿额外各16/16；末段抖动小，暂不改起身权重，非实物或正式大样本验收。起身坐下专家直接组合循环失败，冻结传感器历史27例却收敛，不能归因独立历史自振。已收集四条件62条成功闭环示范（训练23250帧/独立seed留出35000帧），监督修复三档、五档冻结其它关节的膝输出修复及三档角速度连续门控全部完成且丢成功例，未部署；拟合回放更平滑不代表实机改善。新增Upright仅在奖励参考到位后惩罚大倾角，保持过渡动作，3项配置/代价测试通过，但GPU6被其它任务占用、启动前拒绝，smoke/100轮尚未运行。已建立当前线程每15分钟接续检查，资源空闲后再验证与训练。板仍sit-smooth-v1，未发运动/部署/改增益标定。用户持续优化/合格自动部署授权仍有效；证据平级RL artifacts/process-repair-v2-20261010/。

> **2026-10-10 用户授权持续优化/条件部署（最新覆盖）：** 用户离开期间要求效果好直接部署替换，效果不好继续分析优化，并检查倒地起身是否同样振荡；覆盖历史“不自动续训/部署”，不表示无人托稳时自动发实机动作。上一轮失败后新建SitStandProcess-v2：实际EMA两膝目标变化/加速度以0.05rad诊断尺度归一化，过程惩罚−0.15；真实相邻输入平滑权重0.3→3，on-policy保留2→0.2，仿真示范保留10。切换扰动采样0.25→1、reset后首次interval同步新指令；BAM增益每world单次采样替代14关节均值，恢复实际±20% P缩放/±10%反电动势速度缩放范围，仍非HD1910固件D辨识。新任务保留旧任务及v3/61维/EMA0.45/HOME/限位/原始历史；新4项配置/过程/增益检查、先前相关20项及64环境5轮/160步1022次扰动短测通过，归一化逐值不变。GPU6/4096环境另1000轮launcher1899602已启动，未完成；名义/扰动/高低增益回归及过程门后再筛选，不部署失败候选。板上哈希只读核对仍sit-smooth-v1坐站85d7…、起身ec665…。倒地起身CPU两seed×6倒姿×4试验正常48/48、物理扰动+一帧速度延迟48/48，最后3秒两膝单帧p95正常0.069/0.056°、扰动0.085/0.067°，本批未见坐站级振荡，暂不改其权重；非实机运动/正式200例验收。证据平级RL artifacts/process-repair-v2-20261010/。

> **2026-10-10 坐站过渡验证/1000轮最终结果（最新覆盖）：** 独立SitStandTransfer训练和CPU/GPU评估全部结束，job.exit=0/completed.txt存在、launcher已退出；训练26分33秒。新增切换物理俯仰扰动、一帧速度延迟和两膝2秒过程指标，修正CPU trial间BAM阻尼/摩擦残留。服务器严格同初态基线名义28/32、扰动9/16；100/500/999轮名义均28/32，扰动8/16、8/16、11/16。最终多余膝行程增加22.9%，丢失一个原成功例，过程筛选selected=null；GPU robust两seed循环17/32、13/32、16/32、20/32，保持25/32、29/32、29/32、28/32，未正式通过。归一化1000轮逐值保持、核心训练源哈希不变；3项过程门及先前20项检查通过。额外两组gyro一致性监督短实验亦因丢失成功例排除。BAM kd_scale为反电动势速度项，未辨识真实HD1910固件D20，不称实机PD已对齐或唯一根因。没有新部署/实物动作/自动续训，板上仍sit-smooth-v1候选、实物改善待独立验证。详见平级RL artifacts/sit-transfer-train-20261010/README.md与server-train/，本机早期基线不同运行环境不能混配服务器候选。

> **2026-10-10 坐站过渡验证/训练启动（最新离线进展）：** 用户要求新增仿真验证并优化训练。独立SitStandTransfer任务保留v3合同，新增切换0.8秒物理俯仰扰动及2秒过程指标；CPU修正每trial BAM摩擦/阻尼残留后严格同初态循环13/16→扰动6/16，证明新挑战能暴露退化，幅度是诊断假设非实物辨识。20项检查、64环境5轮正规训练/导出及264次扰动短测通过，归一化冻结逐值不变；GPU6/4096环境最多1000轮，launcher1662112已启动，未完成。种子为当前平滑actor，PPO低LR+行为保留/真实相邻输入平滑约束，平滑课程250/500轮生效，完整坐站双向回归后筛选，不自动部署/续训。平台仍sit-smooth-v1候选且实物改善待验证；本次不发动作或改标定/PD/保护。详见平级RL artifacts/sit-transfer-train-20261010/。

> **双向微调最终失败（最新覆盖）：** 2026-10-10 双向种子后1000轮PPO微调及GPU/CPU评估已全部结束，耗时28分35秒；最终权重再次退化，GPU nominal两个重复循环两种子均0/32，CPU两循环两种子均0/8（合计0/32），CPU站保持两组7/8、坐保持两组8/8。GPU门分类：从坐姿起身高度/腿姿几乎全失败；站姿起始下降也有23～24/32失败，多数倾角仍通过。不能把种子31/32写成微调最终结果；初始双向组合与监督种子保留，未替换它们或部署。说明现有PPO目标/优化过程未维持已学双向动作，具体奖励/探索/课程因素尚未隔离证明，不能断言唯一原因。未自动续训、未实物操作；证据平级RL artifacts/sit-contact-bidirectional-20261010/finetune1000/。

> **双向坐站优化（最新覆盖）：** 2026-10-10 双向坐站最新：相同v3合同的两轮专家按姿态指令组合，GPU正常循环96.875～100%、扰动最差78.125%；CPU循环31/32，站保持15/16、坐保持16/16。已用64000条真实组合回放初始化单actor，单actor CPU循环同为31/32，站保持14/16、坐保持16/16；43测试、5轮种子冒烟/导出通过。训练仅新增强制反向采样4～5秒/24秒episode，GPU6隔离1000轮微调launcher1051370已启动，未完成/未验收/未部署，限位/接触/顶压/EMA/PD保持。详见平级RL artifacts/sit-contact-bidirectional-20261010/README.md；旧实物坐下仍失败。

> **坐站第二轮最终结果（最新覆盖）：** 2026-10-10 objective-v2的1000轮及CPU/GPU评估全部结束，job.exit=0/completed.txt存在，launcher已退出；耗时27分36秒。仍未通过：GPU nominal重复循环两种子全0/32，robust仅1/32一组、其余0/32；CPU两重复循环两种子均0/8，站保持两组7/8、坐保持两组8/8。转换窗口表明方向与首轮相反：下降基本成功，但起身窗口height/legPose大量失败，tilt基本正常；CPU循环稳定比例典型[1,0,1]/[0,1,0]。不能把奖励上涨110.81称技能通过，也不能称仅缺训练时长。权重bc2fb7663471f4c9685b55a7b23b770e5db183d5dc3685d49721366390358c3d，未续训/部署/实物动作。证据平级RL artifacts/sit-contact-objective-v2-20261010/{gpu-nominal,gpu-robust,cpu-runtime}.json。

# AGENTS.md

## 2026-10-10 坐站奖励优化第二轮（离线，未部署）

首轮1000及CPU/GPU评估全部结束：CPU重复循环两种子全0/8，站保持7/8、8/8，坐保持全8/8。主要失败是站起后不响应坐下。三项端点审计旧SIT站着3.9348/坐着4.0（非总奖励）；独立候选已改整体腿姿Gaussian、高度归一化/权重3、删除高站姿奖金并延后平滑税。接触/限位/顶压/EMA/61维合同不变。33测试、64环境5轮冒烟/导出及CPU/GPU链路通过后，隔离GPU6从头再1000轮，launcher653594，未完成/未部署。板上旧坐下仍失败；旧证据保留，静态保持及短训练非验收。



## 2026-10-10 坐下实物失败后的候选修正

用户确认顶住仍下压并要求继续。独立SitStandContact-HD1910候选采用自身接触、显式neck_pitch/jaw_soft装配凸包排除、EMA+HOME后XgoDuck范围目标裁剪及顶住误差惩罚，raw历史/61维不变；膝±1.25rad、髋∓0.4579rad、坐高66.442mm。先前±1.2rad CPU短测过而GPU失败，禁止用单一短测宣称通过。新GPU64/CPU32静态保持和5轮冒烟通过后启动独立4096环境1000轮初看，不自动续训/部署；运动循环及实物仍待验收。旧任务/权重/供体运行机制不改写，细节docs/server-training.md最新记录。

## 2026-10-09 用户要求 RL / observer 与 XgoDuck 限位统一

microduck 与 replica 的全部 14 舵机关节物理范围以 `src/mjlab_microduck/robot/xgoduck_joint_limits.json` 为准（2026-10-09 XgoDuck robot_walk.xml 快照）。颈俯仰 −135～60°、头俯仰 ±105°、左髋偏航 −0.22～0.45rad、右髋偏航 −0.45～0.22rad；其余同快照。行走/全碰撞/地面接触/滚轮/回差变体均对齐，CAD 导出须运行 `robot/apply_xgoduck_limits.py`，不改被动关节范围。此要求覆盖旧 MicroDuck/replica 的 −90° 颈限位及旧髋范围，不改变 STL、BAM、HOME、观测或动作合同。observer 软件范围已同值；本次本地改动无实机/EEPROM操作。历史评估对应当时模型哈希，不随新限位改写为已通过。

RL training environments for Microduck — a ~800 g, ~25 cm tall bipedal
local replica with 14 HD1910 policy-controlled joints — built on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo Warp) with PPO (rsl_rl). Policies are trained here at 50 Hz, exported to
ONNX, and deployed by the runtime in the `pollen-robotics/microduck` repo on
the real robot. Sim2real transfer
is the whole point: every convention below exists because breaking it produced a
policy that worked in the viewer and failed on hardware.

## Commands

```bash
uv run list-envs                                    # live task registry
uv run train <TASK_ID> --env.scene.num-envs 4096    # train (add --hf-jobs for Hugging Face Jobs)
uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5   # SMOKE TEST — always run first
uv run play <TASK_ID> --wandb-run-path <entity/project/run_id>
uv run scripts/export.py <TASK_ID> --wandb-run-path <...>   # → ONNX (bakes obs normalizer — mandatory path)
uv run publish --task <TASK_ID> --wandb-run-path <...> --checkpoint N --repo <user>/microduck-<name> --kind episodic --duration-s 4.0
                                                    # → HF Hub repo (policy.onnx + schema-2 manifest.json + README) the daemon loads via `robotctl policy add`
uv run scripts/infer_policy.py --walking out.onnx   # CPU MuJoCo deployment rehearsal (BAM M6 actuators as in training; --no-bam = XML PD)
uv run --with pytest pytest tests/
```

A 5-iteration smoke test at 64 envs catches ~95% of config errors for cents.
Never launch a long run without one.

## Repo map

- `src/mjlab_microduck/tasks/mdp.py` — ALL custom MDP functions (rewards, events,
  observations, commands, curricula). Add new functions here, grouped by task.
- `src/mjlab_microduck/tasks/microduck_*_env_cfg.py` — one cfg module per task
  family. `microduck_velocity_env_cfg.py` is the main walking recipe AND the
  shared base (robot, DR, obs, commands) other envs build on or mirror.
- `src/mjlab_microduck/tasks/__init__.py` — task registration (base + `-Backlash-` variants).
- `src/mjlab_microduck/tasks/backlash.py` — wraps any env cfg into its backlash twin.
- `src/mjlab_microduck/robot/microduck_constants.py` — robot cfgs, HOME frame, BAM actuator cfg.
- `src/mjlab_microduck/robot/microduck/` — MJCF exports from Onshape
  (onshape-to-robot, one `config_mjcf_*.json` per model) + scenes + `add_backlash.py`.
  Collision families: `walk` (feet only), `groundcontact` (curated floor set),
  `allcollisions` (every part; XL330 housings named `*_servo_collision` by
  `name_servo_collision_geoms` → VelStand's servo-impact sensor). Each has a
  `_backlash` twin generated by `add_backlash.py <xml> --backlash-deg 2.0`.
- `src/mjlab_microduck/actuator/friction_dr_bam.py` — BAM actuator + friction DR + backlash encoder.
- `src/mjlab_microduck/export.py` — the ONNX export (normalizer baked in); `scripts/export.py` wraps it.
- `src/mjlab_microduck/publish/` — `uv run publish`: schema-2 manifest builder + ONNX shape/smoke
  gate + Hub upload. Contract = `docs/policy-manifest.md` in the `microduck` repo; only
  constant-command episodic/perpetual policies are publishable (phase/posture-flag are the set's).
- `scripts/` — export wrapper, infer, sim2real comparison, wandb helpers.
- `tests/` — cfg-invariant and mdp-function regression tests (CPU, no GPU needed).

## Invariants — do not break these

- **Obs layout is 61D (actor) and shared across the whole policy family** so
  policies are hot-swappable in the runtime: 48 base proprioception +
  13D command block `[twist(3), head_pose(4), body_pose(6)]`, in that order.
  An env that doesn't use a command slot ZERO-PADS it (keep the obs term,
  sample tiny ranges) — never delete a slot.
- **Joint layout** (14 servos, ctrl idx = joint idx on walk/groundcontact
  models): 0–4 left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle), 5–8
  neck/head (neck_pitch, head_pitch, head_yaw, head_roll), 9–13 right leg.
  On roller/backlash models, passive joints INTERLEAVE — never hardcode joint
  indices in mdp functions; use the `_servo_joint_ids` / `_servo_joint_pos`
  helpers in mdp.py (identity on plain models, correct everywhere else).
- **Unactuated joints are all named `passive_*`** (wheels, backlash hinges).
  Every actuator/obs/reward selector uses `^(?!passive_).*` — keep the prefix
  convention when adding joints, and new `passive_` regexes must not
  accidentally match backlash joints (`^passive_.*wheel`, not `^passive_.*`).
- **Actuators are BAM** (the supplied HD1910 M6 JSON, friction computed by
  the actuator). Two consequences: any STANDALONE env cfg must register the
  `expand_bam_friction_fields` startup event, and joint-friction DR must scale
  the actuator's `friction_scale` — `dof_frictionloss` is zeroed under BAM, so
  randomizing it is a silent no-op.
- **Obs normalization is ON** → the normalizer must be baked into the ONNX.
  `scripts/export.py` does this; in-sim play hides the bug (it applies the
  normalizer anyway), so never hand-convert a checkpoint.
- **Policies are UNFILTERED** (no action low-pass in training). Don't add EMA
  filtering without a matched runtime flag and a transfer test — trained-with /
  deployed-without (either direction) breaks transfer.
- **Domain randomization must not accumulate across resets.** mjlab 1.3.0's
  `dr.*` ops with `operation="add"/"scale"` are natively non-accumulating (they
  re-read compile-time defaults); custom DR functions must restore-then-apply.
  An accumulating CoM randomizer once degraded every long run for months.
- If an obs is remapped to a sensor view (backlash encoder, bias), any tracking
  REWARD on the same quantity must measure the same view — otherwise the policy
  is punished for correcting what it sees.
- `-Backlash-` task variants must mirror their base task's robot model
  (walk / groundcontact / rollers) so backlash A/B comparisons are unconfounded.

## Building a new env — the workflow

1. **Pick the closest template** and build on it, don't start from scratch:
   locomotion → the velocity recipe; episodic trick ending in a pose →
   standup; commanded two-state → sitstand; dynamic maneuver → roulade
   (read its cfg docstring — it encodes a 5-run lesson arc). Building on
   `make_microduck_velocity*_env_cfg` keeps DR / obs / noise / delays in sync
   for free; if you build standalone from mjlab's base template, you must port
   the whole DR + obs-noise + NaN-guard stack yourself (grep for what velocity
   wires: `_safe` critic obs terms, `nan_state` termination with sensor_names,
   `expand_bam_friction_fields`, encoder bias, IMU misalignment).
2. **Verify physics assumptions in sim BEFORE training** — this is the single
   biggest time-saver:
   - A target/rest pose must be a stable equilibrium: hold its ctrl for 3 s
     from noisy inits and check TILT, not just height (a settle test that only
     records z reports fallen states as "resting fine").
   - Measure target heights off the actual robot in sim (e.g. trunk z under a
     standing policy), never carry them across model revisions. A 5 mm-wrong
     STAND_Z once turned the goal into an impossible target for days.
3. **Config conventions**: `ENABLE_*` toggles + tuned constants at the top of
   the cfg file; factory `make_..._env_cfg(play: bool, rough: bool)`; register
   in `tasks/__init__.py` (+ the `_BACKLASH_TASKS` table if applicable); own
   `RslRl...RunnerCfg` with a distinct `experiment_name`. Symmetry mirror-loss
   is available (61D table in `symmetry.py`) — OFF by default, never for
   asymmetric tasks.
4. **Write cfg tests** (see `tests/test_*_cfg.py`): joint indices resolve on
   the actual model, reward weights have the intended sign, gates open/closed
   where expected. These run on CPU and lock in the invariants.
5. **Smoke test** (64 envs, 5 iters): builds, steps NaN-free, obs is 61D,
   every reward term computes, ONNX exports.
6. Train, watch the log (below), and expect 2–5 iterations of reward-hacking
   whack-a-mole — that's normal, the lessons below shortcut most of it.

## Reward design — rules that were each learned the hard way

- **Sign convention (bit four envs):** mdp.py has two penalty styles. mjlab-base
  cost functions return ≥ 0 → negative weight. Self-negating microduck functions
  (`*_penalty`, `*_l1` returning ≤ 0) → POSITIVE weight. A negative weight on a
  self-negating penalty double-negates into a reward for the violation, and the
  policy will farm it (butt-hopping, crash-sits). **The infallible check: on
  every run, every `Episode_Reward/<penalty>` in wandb must be ≤ 0.**
- **RL optimizes the letter of the reward.** Every under-specified degree of
  freedom will be exploited (ballistic whip instead of a roll, shoulder-roll
  instead of sagittal, head-tripod instead of standing). Encode what counts as
  the maneuver in hard state-based gates (support contact, orientation-axis
  checks, latches), not in small penalty nudges.
- **No jackpots:** any "reach X" reward must be rate-limited or slewed.
  Arriving early at a goal state that then pays per-step is a jackpot that
  buys arbitrary violence. For commanded transitions, track a slewed internal
  target (constant-rate blend) — being ahead of the ramp pays zero, so slow IS
  the argmax. Speed-cap penalties alone integrate to a bounded cost and lose.
- **Never gate a positive reward on being in a bad state** (fallen, low) — the
  policy parks in the cheapest qualifying pose and farms it. Use
  potential-based shaping instead (pay Δprogress, e.g. Δcos(tilt): rising pays,
  holding pays zero, unfarmable). For rest tasks, audit each positive term
  against every stable flop (on back / face / side): if flopping keeps most of
  the stack, the policy will flop.
- **Episodic pose-landing tasks:** single fixed target from t=0 (Gaussian + L1
  on joints and height, generous std) + |a_z| impact penalty + two-layer
  upright — NOT keyframe/waypoint trajectories (the policy camps at
  waypoints). The path is what RL is supposed to discover.
- **Regularizers come in two kinds.** Motion-blockers (body_ang_vel,
  angular_momentum, pose std) penalize what a dynamic motion physically
  requires — keep them LOW for dynamic tasks. Smoothness (action_rate,
  joint_torque_rate) damps jitter without blocking slow big motions — safe to
  weight, but introduce it AFTER skill discovery (curriculum from ~0): any
  attempt-tax active while a hard skill is being explored makes "do nothing"
  win. Slow careful tasks (reaching) want heavier smoothness than walking.
- **Compare reward mass, not weights, when copying regularizers between envs.**
  PPO sees relative advantage: the same action_rate weight is 4× weaker under a
  4×-larger positive task stack.
- **Tracking Gaussian std:** ≈ the error you still care about, not the max
  error — too loose has no gradient at small errors. BUT before tightening,
  ask whether the error is escapable by the policy or inherent to the behavior
  you want (a 38%-of-body-mass head MUST oscillate while walking; a tight
  instantaneous head-tracking std taxed walking so hard the policy stood
  still). Price only the escapable part — e.g. L1 on a 1 s EMA charges DC bias
  and lets oscillation cancel.
- **Multiplicative composites beat additive sums at goal states:** when an
  additive stack has a compromise basin (80% of every term via a lean), a
  product of Gaussians collapses on any single deficient factor — but pick stds
  wide enough that the CURRENT policy scores visibly, or the gradient is
  invisible and nothing changes.
- **Joints parking on hard limits:** fix with a qpos-side limit-proximity
  penalty on the offending joints; the stock `dof_pos_limits` only fires in the
  last ~7.5% of range, and command-side penalties don't work (wide ctrlrange is
  intentional — low-kp servos need overshoot).

## Commands, observations, dead weights

- **A command input that is never non-zero has dead weights forever.** Every
  command slot keeps a small non-zero sampling range from step 0 (even at
  reward weight 0) so its input neurons stay alive for later curricula.
- **Zero-command behavior must be explicitly trained** (`zero_command_prob`-style
  exact-zero sampling): uniform sampling essentially never produces the all-zero
  command, which is exactly the deployment idle state.
- Rare-but-important command regions need explicit buckets — e.g. turn-in-place
  (`rel_turn_in_place_envs`): independent uniform sampling made spinning ~2% of
  experience and it never trained.

## Curricula

- Steps are env steps: `iteration × 24` (`NUM_STEPS_PER_ENV = 24`).
- Use the proven split: `microduck_mdp.reward_weight` for weight schedules, a
  dedicated params-curriculum for command/event ranges. `mdp.reward_weight` is
  a step function, not an interpolation — discretize ramps into stages.
- Mutate term cfgs via the managers (`env.event_manager.get_term_cfg(...)`),
  never `env.cfg.events[...]` — managers deepcopy their cfg at init, so writes
  to `env.cfg` are silent no-ops (this also bites eval scripts that force
  spawn states).
- **Phase-align every stage with what the policy has actually learned**: don't
  harden spawn mixes before the current slice consolidates; don't introduce
  taxes before the skill exists. When a wandb metric steps DOWN exactly at
  curriculum stage boundaries, the pacing is wrong — stretch stages or move
  the introduction later, never earlier.
- Reverse-curriculum spawns (starting episodes partway through the maneuver,
  including nearly-done) are the reliable fix for "learns the start, never the
  last mile" — the frontier otherwise gets no on-policy data.

## Training ops & reading a run

- wandb project `mjlab_microduck`; logs in `logs/<experiment_name>/`; resume
  with `--agent.load-checkpoint model_XXXX.pt --agent.resume True`.
- **Warm start ≠ resume.** mjlab's runner stores `common_step_counter` in the
  checkpoint and restores it (plus the iteration) on load, so loading another
  task's checkpoint jumps every step-based curriculum to its final stage in
  iteration 1. Prefix `MICRODUCK_WARM_START=1` (mdp.py Patch 5, forwarded to HF
  Jobs) to restart both at 0 while keeping weights/normalizer/optimizer. Also
  collapse the inherited curricula of the SOURCE task to their final stage in
  the target cfg (see `_collapse_curricula_to_final` in velstand) — the loaded
  policy was trained under those conditions. Only warm-start across tasks with
  compatible normalizers (the stand expert's twist std is 0.005: never seed a
  walking env from it).
- Deployed-policy provenance: Hub ONNX files carry `run_path=None`; find the run
  by exact last-layer weight match against wandb checkpoints (2026-09: walk =
  441tzs6d@3750, stand = 69u48n8l@9750).
- Watch per-iteration: mean reward rising AND episode length behaving as the
  task demands; every penalty term ≤ 0; the MAIN task term actually growing
  (total reward can rise purely on regularizers while the trick never happens).
  `Episode_Reward/<term>` logs the WEIGHTED value — a term at weight 0 reads 0
  regardless of behavior, so interpret against the weight schedule.
- Budgets: simple episodic tricks ≈ 1000 iters at 4096 envs; gaits and
  curriculum-heavy recovery need 4000–6000.
- **Measure before theorizing.** When a run "fails", run a headless eval of the
  actual checkpoint (per-spawn-type batteries, end-state clusters, angular-rate
  profiles) before changing rewards: past "failures" turned out to be early
  checkpoints, a success criterion splitting one behavior cluster in half, and
  a pay cap fighting measured physics. Sim metrics can pass while the video
  fails the human eye — watch the video AND check which geom/axis touches.
- Report what rollouts actually show ("rolls but face-plants 1 in 3"), not
  "it works!". The user decides when it's good enough.

## Sim2real footguns (cost real debugging weeks)

- A fresh `uv sync` is the ground truth (HF Jobs run one): anything that only
  works via manually-installed local packages will die remotely. Keep
  `pyproject.toml` honest.
- **Wheels are per-architecture.** On linux-`aarch64` (DGX Spark / GB10) PyPI's
  torch wheel is CPU-ONLY (`2.9.1+cpu`, `torch.version.cuda is None`), so
  `torch.cuda.device_count() == 0` and mjlab's `select_gpus()` indexes an empty
  list → `IndexError` before iteration 0. `[tool.uv.sources]` routes torch to
  the cu129 index for `aarch64` only (cu129 matches the CUDA toolkit warp
  bundles; x86_64/HF Jobs stay on PyPI). Two silent break points, both locked
  by `tests/test_aarch64_cuda_torch.py`: torch must stay a DIRECT dependency
  (uv applies `[tool.uv.sources]` to direct deps only — deleting the
  redundant-looking `torch==` pin makes the routing a no-op), and the pin must
  stay `==`, since the CUDA index carries newer builds than PyPI (a `>=`
  silently dragged torch 2.9.1 → 2.13.0).
- Physics-aligned limits: a 25 cm robot tumbles at 3.5–5.5 rad/s NATURALLY —
  don't impose human-scale speed intuitions via caps; put anti-violence
  pressure on impacts and thrash (|a_z|, action_rate, support gates), not on
  rotation speed.
- IMU DR is zero-centered — it trains tolerance to misalignment magnitude, and
  CANNOT compensate a systematic mounting bias (that's a runtime calibration).
- Real deployments hot-swap ONNX policies (walk / stand / trick) with a shared
  obs contract — rehearse in `scripts/infer_policy.py` before touching the
  robot, with the correct command-slot writes (a posture flag lives in the
  twist vx slot; feeding all-zeros means "stand", which looks like "policy
  ignores the button").


## 2026-10-04 用户明确要求XgoDuck部署对齐

HD1910 stand/recovery/settle任务新增XgoDuckPositionActionCfg：动作EMA alpha0.45，raw action保留给manager观测，reset支持部分环境。此为用户要求的模型特定例外，其他任务继续保持UNFILTERED。部署原生平台同样alpha0.45/scale1/原始prev_action；新训练必须保留匹配配置，不能导出后省略运行端EMA。当前v5权重没有重新训练，不能把配置修改称为新模型验收。HD1910噪声范围按XgoDuck actor gyro±0.06、gravity±0.03、joint_pos±0.01，脚摩擦0.4–1.5、训练vin7.4–8.0；vin_min4.0与本机用户电压规则保持。实物ID/零点/IMU和本机几何/HOME不改成对方机器的值。
