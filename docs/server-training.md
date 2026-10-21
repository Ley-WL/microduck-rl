> **2026-10-21 当前采用：** 官方配方 + HD1910 + Observer机械/目标限位，从头5000轮；最终模型e3b90ac0保存于 `models/walk_observer_hd1910_5000/`，用户实机反馈效果很好。训练配方、独立30输入评估及限制见 [说明](walk-observer-official-training-20261020.md)。10000轮已完成但未全面超过本版本，不替换现役。

> **2026-10-14 大幅舞蹈 v5 抗扰续训已启动：** 用户要求加强动作与抗干扰。v4 已完成800轮，正常64/64、训练式扰动55/64、周期压力57/64，尚未通过抗扰验收。独立 `dance10-resilient-v5-20261014`（launcher PID122660）载入v4 final，保留完整大幅动作且取消幅度渐增；训练水平推扰±0.18m/s、间隔1.5–4秒、每回合24秒连续三轮。HD1910/当前STL/61维/EMA/机械限位/45度终止门保持，actor归一化冻结、Adam重置、std .08、LR1e-5、seed121031。12项本地测试通过，64环境×5轮及带归一化ONNX导出通过，已进入1024环境×1000轮阶段。结束自动按v4原评估条件/两旧seed对照，再用120011、120017各64环境做新旧配对压力评估，核验初始qpos/qvel/观测哈希及实际动作幅度，并留视频复核；不宣称全部物理随机量一致。查看evidence/progress.json与job.exit，勿重复启动。未部署或访问主板，仿真非实机验收。详见docs/dance10-training.md。

> **2026-10-14 用户要求舞蹈更魔性/大幅v4启动准备：** v3训练及回放已完成：正常128/128，旧stress63/64，新seed62/64、63/64，仍有3失败/2旧成功退化，未完全验收或部署。用户观看后明确要求放大动作。新增独立Dance10Expressive，头偏航/pitch及身体侧摆2倍、歪头/屈膝2.5倍（下沉目标15mm），前300轮渐增/后500轮全幅；加强body跟踪、降低固定腿姿奖励，HD1910/STL/EMA/61维/终止门保持。11测试通过，目标头部加HOME距限位≥.05rad；小幅/全幅参考各32/32无倒地，实际头偏航30.43→60.63度、body roll5.37→12.05度、高度7.33→20.19mm，未当抗扰验收。64×5/导出已通过，800轮阶段在途。原v3权重固定作种子与小幅/全幅参考；64×5/导出通过才1024×800，结束两新seed三条件全幅回放及实际幅度/视频，禁止用小幅课程结果当大幅验收。独立dance10-expressive-v4-20261014 launcher3979552，首次上传超时仅空目录、重传后启动，无旧作业重启或实机操作。查看progress/job.exit勿重复启动。详见docs/dance10-training.md。

> **2026-10-14 舞蹈v2优化完成/v3恢复训练启动：** v2正常/训练扰动128/128仍通过，摆头/点头幅度相关性通过，头部误差降约30–40%；stress46→54/64，但丢原6例，accepted=false。100/200/300检查点44/51/55且丢原11/9/6，未解决退化。v2完整证据SHA557124a8381bd6dfda803e20cb29af4d4397a6daff6a0ddf419617b4f19349ea下载核验、源hash保持。独立v3 Dance10Recovery扩大初态/推扰范围，保持节拍幅度/HD1910/STL/EMA/奖励/终止角门，载入v2 final，actor normalizer冻结/Adam重置/std.08/LR1e-5，先64×5再1024×400及五组回归/两新seed100003、100019的原v1配对压力筛选。11测试通过；新服务器目录dance10-recovery-v3-20261014，launcher3434233，查看progress/job.exit勿重复启动。当前没有已通过的抗强扰动新候选，未部署/访问主板，仿真非实机验收。详见docs/dance10-training.md。

> **2026-10-14 舞蹈稳定性优化（进行中）：** 用户要求继续优化倒地。冻结v1 model999实际回放：nominal/robust×seed81371/91373各32例合计128/128完整16秒未倒，不能将探索训练日志当成策略失败率。诊断stress（初态roll±.08/pitch±.10、关节±.04，每3秒水平速度推扰±.16）仅46/64；18失败中17例在推扰后.7–1.2秒。新独立Dance10Stable保留节拍/幅度/HD1910/STL/EMA/观测/终止角门，修复zero-twist打滑惩罚失效、加超额倾角/偏航代价及恢复练习。actor/critic加载v1、Adam重置、actor normalizer冻结、std.10/LR3e-5；smoke64×5/导出通过，1024×400在途，launcher3319271位于新服务器目录dance10-stable-v2-20261014，不重复启动。10项本地测试通过，结束按固定seed/配置做新旧逐例存活和幅度回归；原评估未存完整reset物理状态，勿称初态逐数组已核验一致。未宣称修好，不自动部署/访问主板。详见docs/dance10-training.md及artifacts/dance10-optimize-20261014。

> **2026-10-14 十拍舞蹈首轮准备/启动：** 用户明确要求着手训练约10拍魔性鸭子舞，授权本次独立服务器训练。新增 Dance10-HD1910：100BPM/6秒舞蹈+前后各1秒站姿、61维指令观测、当前replica与HD1910/P5/EMA旧权重.45；从头训练、无官方教师。11项本地检查通过。服务器已迁移到 `/CodeRepo/Code/leiwu30/1_code/microduck-training`（已核对迁移记录），旧home路径失效；本轮独立 `dance10-v1-20261014` launcher2705983，32世界3秒站姿检查无重置/最大倾角6.2度、64×5冒烟及归一化ONNX导出通过，2048×1000已真实迭代（详见本地launch-status.json）。以 evidence/progress.json/job.exit 和日志为准，不重复启动；尚未动作验收，不自动续训或部署、不访问主板。详见 `docs/dance10-training.md`（本记录位于docs时使用同目录文件）。

> **2026-10-14 本地训练配置整理：** 用户要求全面统一 HD1910 与当前 STL。所有69个注册MicroDuck任务/138个train-play配置使用本地replica和1910 M6；删除共享XL330/P200/6V默认、默认官方教师、迁移工厂供体惯量覆盖和当前MJCF旧PD模板。STL/装配/质量惯量/关节范围逐数组保持；64输入文件SHA清单，185检查通过/2跳过（旧warm-start关闭、GPU缺CUDA）。仅本地源码整理，未同步服务器、未重复/终止已有任务、未训练/部署/访问主板；历史远端结果仍对应当时源hash。详见[硬件整理记录](hardware-profile.md)。

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

> **2026-10-12 坐姿起身磕头优化（当前，无实机动作）：** 用户确认磕头来自坐姿站起，当前倒地起身4b9c82e5已用户成功反馈，保持板权重。App0.9.43已OTA并取消坐下预备/站起，只用倒地起身启动。observer代码/新模型cfe374a、App463f730已推送；RL4a1d7bf仅本地提交，用户明确不建远端、不推上游。当前坐站1c94934c无EMA/现行机械范围CPU新回放正常8/8无头颈接地；扰动+1帧速度延迟6/8，失败两例出现头颈接地（模拟法向力峰55.78/54.25N，不是实机力测量），其中一例首秒倾32.08°且接地；回放连续坐站，没有重放板1秒回行走，非完整实机因果证明。原训练允许头辅助，不等于其唯一根因。完整16例CPU归档SHAe376b5e767f59ce79e36cc9fa081371fb8ec1947de131d68ba25b3e33bcdd9cc下载核验解压，RL artifacts/sit-head-clearance-20261012/probe。

> 独立sit-head-clearance-20261012新小预算对照launcher1036540（v3）运行中，仅GPU0，GPU1他人高占用不动；两个分支共用当前actor映射/机械止挡+HOME+编码/noEMA/raw历史/Feetech BAM/观测/critic，区别为站起指令下头颈接地软代价0或−1。种子由当前ONNX恢复权重及归一化mean/std，128冻结观测Torch/ONNX差2.146e-6rad；critic及normalizer count借自历史HD1910坐站checkpoint，并非供体原critic/count，不能把此假设当已校准；两组共同探索std.05/lr初值1e-5/adaptive调度/trainseed61723/5epochs×4minibatches。2配置/代价检查通过，先各64×5smoke正规导出/finite/normalizer/Adam步100，后从同seed分别1024×20/Adam400，再相同8正常+8扰动循环；不是正式验收、候选不自动部署。首驱动误假设每轮8更新，实际20，已完成Control smoke复用不重训；第二次导出审计主进程误载全局任务注册表，改为显式独立src，源/日志保留attempt1/2，当前v3不重复旧训练。旧job.exit1属于前次退出，跟踪launch-v3 PID及当前progress，不只读旧退出标记；后续最终检查sourcehash、完整相同初态/成功保留/头接地/过程/frame再决定扩大。主板/标定EEPROM/PD/BLE/其它技能及网页不改，仿真非实机验收。

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

# 服务器训练

## 2026-10-10 坐站奖励优化第二轮（离线，未部署）

首轮1000及CPU/GPU评估全部结束：CPU重复循环两种子全0/8，站保持7/8、8/8，坐保持全8/8。主要失败是站起后不响应坐下。三项端点审计旧SIT站着3.9348/坐着4.0（非总奖励）；独立候选已改整体腿姿Gaussian、高度归一化/权重3、删除高站姿奖金并延后平滑税。接触/限位/顶压/EMA/61维合同不变。33测试、64环境5轮冒烟/导出及CPU/GPU链路通过后，隔离GPU6从头再1000轮，launcher653594，未完成/未部署。板上旧坐下仍失败；旧证据保留，静态保持及短训练非验收。



## 2026-10-10 坐下堵转修正候选（实物失败后的独立训练，未部署）

**1000轮训练已完成，回放未通过（最新覆盖）：** 检查点999按1000次迭代结束，训练PID435664已退出；导出ONNX SHA256 c6498ebfe6c88b9980cafd23dc337a8d89555419fa0a78bf1f53fde750c6dd89。GPU nominal/robust两种子×四场景×32世界全部完成且accepted=false。nominal两循环分别seed1009为3/32、4/32，seed2027为2/32、2/32；两保持全部32/32。robust两循环为8/32、9/32、4/32、1/32；站保持31/32、32/32，坐保持29/32、28/32。保持明显优于重复转换，尚不足以更新实物。查询时CPU runtime回放仍执行，首组stand_cycle0/8，launcher435658是评估脚本存活，不代表训练继续。未自动续训或部署。完整GPU结果本地artifacts/sit-contact-repair-20261010/firstlook1000/。

**第500轮中途初筛（不是验收）：** 在独立空闲GPU4导出500检查点，GPU nominal单seed1009每场景16世界：站姿保持16/16、坐姿保持16/16，站起点循环0/16、坐起点循环0/16。相同ONNX本地CPU runtime单seed每场景2次：两保持各2/2、两循环各0/2。抽查轨迹从坐姿能站起，但再次要求坐下仍保持约115mm站高，未到约66mm坐高；当前主要失败为不执行下降而非可据此证明实物不顶压。新限位/接触条件下已经学会保持，双向切换未学成；继续原1000轮，不中途改奖励/重启/部署。证据artifacts/sit-contact-repair-20261010/probe-500/；500模型SHA256 c5da9092740f75bb77b99723f72ecc8f8f1a032e117cd14cfa3d4f41a6b73c1d。不能以小样本保持100%宣称模型实机可用。

用户确认实物腿/头顶住仍下压，要求继续修正。新增独立任务 `Mjlab-SitStandContact-Flat-MicroDuck-HD1910`；旧起身/坐站/行走任务及已部署权重不改写。候选启用68身体网格及双脚的自身接触（contype3/conaffinity2，地面bit1/自身bit2），关闭重复碰撞网格；CAD HOME到坐姿扫描发现neck_pitch/jaw_soft凸包恒定约2.1mm重叠，显式排除这一装配对，其他新接触不自动排除。实际顶住部位及CAD凸包对应真实表面的精度仍待实物核对。

训练目标在EMA0.45/0.55 + HOME之后按14关节XgoDuck物理范围裁剪，raw prev_action仍保留；CPU回放复用接触合同及相同目标限位，Rust仅识别新v3元数据时启用同范围裁剪再换算编码/EEPROM饱和，原供体及旧模型处理保持。新模型元数据缺14项范围或不一致会拒绝；不加实测角度越界卸力、负载门槛，不修改4.0～8.4V保护/标定/增益/EEPROM。

原坐参考膝±1.35rad在完整接触下目标误差较大；候选±1.2rad CPU16次3秒保持全通过，却GPU较长5秒只27/64直立，记录这项失败，不作为长训依据。进一步扫描12组膝/髋组合，选膝±1.25rad≈71.62°、髋俯仰∓0.4579rad，GPU64次带小扰动5秒全直立，CPU32次5秒全直立；坐高GPU中位66.442mm，CPU均值66.074mm。GPU完整环境保持检查：clean坐33/33、带原reset扰动坐31/33，站31/31和30/31；这是静态参考检查，非模型循环成功率或实物验收。候选同步更新坐姿奖励/命令高度/重置高度，增加raw目标越限惩罚和已限位目标与反馈的误差惩罚，避免合法角度但顶住仍持续追目标。

32项RL/回放测试通过；Rust93通过/1忽略，ARM构建通过。修订候选64环境5轮冒烟exit0、nan_state0、新惩罚实际为负；61→14归一化ONNX导出、元数据及CPU四场景×两种子/GPU短回放链路有限值完成，5轮模型成功率0%，仅工具链检查。未部署/发运动。

按用户此前1000轮先看效果要求，GPU6从头启动4096环境1000轮有限训练，隔离源码 `/home/leiwu30/microduck-training/sit-contact-repair-20261010`，launcher435658；不是续训旧发散权重。作业结束自动导出及GPU nominal/robust（32环境×两种子）和CPU runtime（每场景8次×两种子）诊断/视频，样本低于正式200×两种子要求，不标accepted。脚本不自动续训/部署。启动不是完成，最终日志/进程/模型回放另行核验。证据本地 `artifacts/sit-contact-repair-20261010/`；核心源码包SHA256 cd8515c5539b2295821861a41c66ca1baf210d1175caee451b305557eb0f6675，启动器后续单独上传，job报告记录执行脚本SHA256。

## 2026-10-10 倒地起身与坐站模型重新训练完成（完整仿真验收）

最终选择起身v3 checkpoint4750和坐站v2 checkpoint5000。两者均使用当前replica STL/XML、HD1910 1910_m6舵机参数及XgoDuck限位；服务器和本地的63个机器人源文件、舵机参数、导出权重哈希一致。每种验收均使用seed1009/2027，各场景每种子200例，同一ONNX权重执行GPU nominal、GPU robust和默认CPU runtime；不是训练奖励/小样本probe达标。

| 模型 | 覆盖 | GPU nominal最差组 | GPU robust最差组 | 默认CPU最差组 |
|---|---|---:|---:|---:|
| 起身4750 | 俯卧、仰卧、左倒、右倒、斜倒、倒置 | 100% | 99.5% | 100%（2400/2400） |
| 坐站5000 | 站姿起始循环、坐姿起始循环、站立保持、坐姿保持 | 100% | 97% | 96.5%（1577/1600） |

全部场景、两种子独立满足≥95%。CPU逐例有限且resetCount=0，GPU成功样例无重置；姿态/高度/速度/腿姿态的稳定窗口判据与完整测试源码一致。20段正式CPU首例视频全部审阅，起身头颈正常展开，坐站切换和保持正常；全部2400条起身CPU回放终态3秒头颈最大HOME误差组均值0.12024～0.12209rad，最差单例窗口均值0.13012rad（约7.46°），辅助统计不替代成功率。两ONNX观测归一化已烘焙，真实本地CPU推理各9组输入形状61→14/有限输出通过。

最终包：`artifacts/hd1910-skills-20261009/hd1910-getup-sitstand-qualified-20261010.zip`（SHA256 72151648805854e8ee428c74f2929b7cf866aabe6ceec3fe6e60a6897941213b），解包目录`qualified-models/`含两个ONNX/元数据、GPU/CPU逐例报告、20段视频、来源核对和SHA256SUMS.json。36个文件与ZIP内容逐一哈希验证通过，ZIP CRC检查通过。`candidate-models/formal-qualification-audit.json`基于真实记录和已审阅视频全部qualified=true；不是直接信任accepted字段。

达标与本地打包完成后，仅向本任务记录并重新核对身份的4个训练PID、8个watcher及2个其子评估发SIGTERM；后续ps确认所有这些PID退出、GPU列表无本任务训练进程。检查点/日志/源文件全部保留，无删除或重启其他GPU作业，停止证据`skills-qualified-owned-stop.json`。仅停止多余训练，不关闭服务器。服务器最终报告已纳入同模型完整验收证据，goalAchieved=true；当前本地状态同步该完成结论。用户未要求实机部署，本轮没有主板访问、控制指令或提交推送；本结论为仿真达标，不宣称实机成功率。

## 2026-10-10 坐站5000完整CPU验收通过，起身4750完整GPU通过

本轮重新读取服务器实际进程与评估文件。选定v2坐站5000的同一ONNX（SHA256 fa0c9c4ab8f2c6a2f096986f11c102a7561c036ea16f6a7069dcf577500e8209）已完成nominal、robust和默认CPU runtime三套验收。每套均stand_cycle/sit_cycle/stand_hold/sit_hold×seed1009/2027×200例；GPU nominal最低100%、robust最低97%。CPU八组依次为seed1009：97%、100%、98%、100%；seed2027：96.5%、100%、97%、100%。本轮独立逐项核对1600条CPU记录，trial编号完整、每条finite=true/resetCount=0，并从success布尔值重算各组成功率；GPU逐例success数组/计数/成功样例无重置核对通过。CPU配置为200Hz implicitfast、50Hz模型、15ms目标队列、0额外观测延迟、0.4秒settle、速度滤波旧0.16/gyro旧0.25、机身freejoint角速度、73活动接触geom；未用诊断变体充当验收。

坐站正式CPU的两种子×四场景共8段首例视频已下载审阅。四循环在0/4/6/10/12/16/18/19.9秒能看到连续站坐转换，四保持在0/4/10/14.9秒姿势稳定、头颈正常。还读取全部1577条成功CPU回放的NPZ，对各稳定窗口的头颈四关节最大HOME误差作统计：组均值0.02957～0.04463rad（约1.7～2.6°），最差单例窗口均值0.10025rad（约5.74°）。此头姿态统计是辅助诊断，不能替代动作成功率或把样例视频解释成所有实物姿态已验收。证据checkpoint-5000/sitstand/{cpu-formal-summary.json,all-head-audit.json,cpu-video-review/{cycle-review.png,hold-review.png}}。

选定v3起身4750（SHA256 ec665f3121d8d45da0abd3d5792955d0aa46ba39f2897e3bbe01788fda9770f4）的完整GPU nominal与robust均完成：六倒地姿势×两种子×200例，nominal均100%，robust最低99.5%（prone2027与right_side2027各199/200，其他200/200）。同权重默认CPU worker2878908现场确认运行中；最新完成seed1009的prone/supine/left_side/right_side/oblique五组，各200/200，剩余尚未验收，不宣布目标完成。原训练仍运行，不改其参数、不重启评估。

本轮后续读取起身CPU新增完成inverted1009，因此第一种子的六姿势1200/1200全成功；worker2878908重新确认运行中，第二种子仍待完成。六正式首例视频已审阅0/3/8/14.9秒，均能从对应倒地姿势起身、头颈正常。对当前六组1200条成功回放计算终态3秒头颈最大HOME误差，组均值0.12029～0.12209rad，最差单例窗口均值0.12923rad（约7.40°）。证据v3/checkpoint-4750/getup/{all-head-audit.json,formal-cpu-seed1009-partial-review.png,formal-cpu-inverted1009-review.png}；统计scopeComplete=false，不代表第二种子已验收。

进一步进度：起身4750默认CPU已完成10/12组（第二种子新增prone/supine/left_side/right_side），全部200/200；第二种子的这四段正式首例视频也已审阅、头颈正常，斜倒/倒置尚待完成。服务器实际读取的模型SHA与本地相同，63文件源/舵机参数/技能factory源SHA逐项核对；原检查点哈希起身4750为578ef3a2cadec4db306bed47a3a747082e8874cd3b985bc3d96f6cc1795ec944、坐站5000为234dfc039f49c4799e650c9b815167bf62987a966d73ff6afb3f8d3502408cc5。两个ONNX图均以Sub/Div开始并含obs_normalizer._mean初始值，本地CPUExecutionProvider各9组输入推理有限且61→14形状匹配。证据candidate-models/{server-provenance-audit.json,local-onnx-contract-audit.json}；坐站5000-export.log确认原model_5000.pt正常导出。最终打包脚本要求全部三验收、两种子所有正式样例视频、上述合同均通过才允许执行，目前未打最终包。

两个选定候选的63个replica XML/STL源哈希全部与当前本地文件一致，HD1910 1910_m6.json SHA256 ef2d51adfb1cc0831b9b02ca19aa9176fceecdf725148d084378d7d9a64afceb一致；61→14、EMA0.45/0.55、无XML目标裁剪和68机身落地mesh元数据检查通过。独立证据审计见candidate-models/formal-qualification-audit.json；qualification-status.json更新坐站完整CPU/8段视频标记，goalAchieved仍false。仅服务器训练和仿真，无实机访问/部署，未提交推送。

## 2026-10-10 坐站第5000轮GPU完整通过，CPU验收开始

上轮progress（4750起身头颈正常视频审阅并启动完整GPU验收）。本轮原版v2坐站5000完整nominal八组均200/200；完整robust seed1009 stand_cycle200/200、sit_cycle196/200、stand_hold200/200、sit_hold196/200，seed2027 stand_cycle197/200、sit_cycle194/200、stand_hold200/200、sit_hold198/200，全部≥97%，两个scopeComplete=true/accepted=true。证据checkpoint-5000/sitstand/gpu-{nominal,robust}-summary.json，policySha256=fa0c9c4ab8f2c6a2f096986f11c102a7561c036ea16f6a7069dcf577500e8209。候选选择以完整结果为准，不强制用早期30秒回合v3：该v3的4250 robust失败，不拿方案名称代替验收。

v2坐站CPU watcher2667839已对5000同一ONNX开始200例×两种子正式runtime回放，初期完成seed1009 stand_cycle194/200=97%、sit_cycle200/200，其余待完成，不能从GPU通过或两CPU场景推断完整CPU验收。4750起身完整nominal六姿势×两种子均200/200=100%，robust还在执行，worker2801547仍存活。所有原训练/候选继续，无实机操作。

两个待验收候选已下载本地candidate-models/getup-4750/getup.onnx（SHA ec665f3121d8d45da0abd3d5792955d0aa46ba39f2897e3bbe01788fda9770f4）和sitstand-5000/sitstand.onnx（SHA见上）；每个与metadata SHA逐字节匹配，本地真实ONNXRuntime CPU校验[1,61]→[1,14]及有限输出通过。qualification-status.json明确formalCpuCompleteAccepted=false/goalAchieved=false，不是已发布或实机验收包；最终确认仍需同权重全部GPU、CPU和视频。模型更新后的STL/HD1910物理源保持不变。

## 2026-10-10 v3第4750轮头颈恢复正常，追加完整验收

上轮分类progress（4500默认CPU96/96及视频头颈改善、4250坐站CPU64/64）。本轮原4000起身正式CPU已完整完成，六姿势×两种子×200例共2400/2400成功，scopeComplete=true/accepted=true；policySha256与两项GPU完整验收一致。因此原4000三个数值profile均通过，但严重折颈的视频失败仍存在，不能据此完成整个目标。证据checkpoint-4000/getup/cpu-formal-summary.json。

v3起身4500完整nominal各场景95%～98.5%，通过；robust早期俯卧/仰躺/左侧93.5%/92%/91.5%，仍未达到门槛（其余当时未完整，不声明完整结果）。v3坐站4250完整nominal最低96%，robust完整两循环seed1009为77%/83.5%、seed2027为87.5%/87%，保持98.5%～99.5%，未通过。早期30秒回合候选尚未改善扰动成功率，继续其预算并比较后续检查点，不挑高分场景代替循环。两种子两循环4个视频0/4/6/10/12/16/18/19.9秒帧已审阅，重复切换确实发生、头部正常；此视频只证明样例姿态，不覆盖200×两种子扰动总体。证据v3/checkpoint-4250/sitstand/{gpu-robust-summary.json,robust-gate-diagnosis.json,cycle-review.png}。

v3起身4750独立导出回放worker2783782实际结束，六姿势seed1009各1例全成功。终态3秒最大头颈HOME误差均值0.121583rad（约6.97°），六例0.120049～0.123585rad，比4500的1.16897rad及原4250的2.708rad继续明显下降。0/3/8/14秒视频帧已检查，颈部正常展开、头部基本水平，属于首个视觉正常的起身候选；这仍是六次样例，非95%成功率。证据v3/checkpoint-4750/getup/{cpu-runtime,head-posture.json,pose-review.png}。

对4750立即启动独立完整GPU验收worker2801547：准确复制已审阅同一ONNX/metadata/provenance到canonical checkpoint-4750，并逐字节核对，nominal与robust各6姿势×2seed×200例，原验收门槛保持。已有CPU watcher2667841仅在两GPU profile都通过后执行默认200×两种子CPU；GPU4500早期robust已低于95%，不会靠其CPU样例取代4750验收。手动完整验证另列additionalValidation，避免report只按probe忽略4750。原版和两候选仍运行，未改活动奖励或部署。

## 2026-10-10 v3第4500轮头颈明显改善，尚未完成验收

上轮为progress（两项优化续训真实启动、修复CPU EGL监测）。本轮核实原训练2052833/2052857、候选2647925/2647957及正式CPU评估2667834均存活。4250原版起身追加64世界15秒审计显示，站稳终态颈目标EMA为−2.546～−2.644rad，全部末3秒样本低于原版物理下限−2.356rad，超量平均0.189～0.288rad（约11～16°）；颈探索std0.138735rad。动作仍未XML裁剪，关节因真实模拟物理限位维持折颈，此结果支持输出饱和是早期逃离困难的一个因素，不证明唯一原因。证据checkpoint-4250/getup/head-target-audit.json。首次审计在结果聚合发生NumPy布尔索引广播错误，未写结果；修正后使用新诊断编号，保留失败日志，未重启训练。

v3起身4500默认CPU回放六姿势×两种子×8例全部成功（96/96，scopeComplete=true，样本不足故accepted=false）。最后3秒四头颈关节最大HOME误差均值从原版4250约2.708rad降至1.16897rad，96次范围1.14353～1.19996rad；颈均误差1.16897，头俯仰/偏航/横滚分别0.11951/0.09330/0.09937rad。12个视频0/3/8/14秒帧实际审阅，颈部已明显展开，但离目标姿势仍较远，不能宣布最终正常头颈通过。证据v3/checkpoint-4500/getup/{cpu-runtime-8probe,head-posture.json,pose-review.png}。此进展撤销“可能需要头部重初始化”的提前假设，继续当前headmax候选，不重置已学恢复。

v3起身4500 GPU初筛六场景均≥95%（俯卧/仰躺/斜倒63/64，左侧62/64，右侧/倒置64/64），已进入正式200×两种子nominal/robust；不能把初筛替代正式成功率。v3坐站4250 GPU初筛四场景64/64，默认CPU两种子每场景8例亦全通过（64/64、accepted=false），正式GPU正在执行。4000坐站robust失败逐端点统计显示坐下高度/腿姿态占主导，站起端点大多通过，记录robust-gate-diagnosis.json，门槛失败数重叠不可相加。原4000起身正式CPU此时前7组各200例100%，其余未完成，不填写推测成功率。最后实查候选4610/4411轮，全部目标继续未完成，无实机部署。

## 2026-10-10 v3独立优化续训已实际运行

上轮分类progress（4250视频审阅、独立headmax冒烟、4000完整nominal证据）。本轮4000 robust完整：起身6姿势×seed1009/2027每场景200例全部100%，scopeComplete=true/accepted=true；坐站两种子stand_cycle分别84%/89%、sit_cycle88%/89.5%，保持98%～99%，scopeComplete=true/accepted=false。两项gpu-robust-summary.json已下载，常规通过不能覆盖坐站扰动失败或起身折颈。

准备独立源microduck-rl-skills-v3-20261010，仅技能工厂和新增hd1910_candidate_rewards.py有配方差异：起身终态头颈EMA惩罚改最大关节误差，坐站回合15→30秒；STL、XML限位、HD1910参数、动作EMA/观测及其余奖励/DR均从当前v2逐文件复制并哈希核对。完整哈希清单skills-v3-source-manifest.json与两个修改文件已存本地source-v3。v2快照未含项目清单，首次准备在复制清单时报错，确认已有src/scripts逐字节完整后恢复准备，依赖继续使用原venv；未改原训练。确切v3源两任务64世界/5轮冒烟均exit0、nan_state0。

首次续训v3-6250/v3-6000因resume布尔选项需要显式True，在CLI解析退出（exit2），未开始训练；失败日志和记录保留，未重复准备种子。修正后独立运行hd1910-getup-v3b-6250（train2647925，launcher2647923，GPU4）与hd1910-sitstand-v3b-6000（train2647957，launcher2647955，GPU6）已实际输出4265/6250及4015/6000。起身种子4250、坐站种子4000；通过prepare_resume_checkpoint保留actor/critic、归一化、Adam及common_step_counter102024/96024，下一迭代4251/4001，各1999次新增更新，总末轮6249/5999，MICRODUCK_WARM_START=0。为共享自己的GPU且不影响无关任务，候选并行环境2048，原版4096训练继续；批量也变化，不能称严格单因素因果AB。新增GPU验收监测2647936/2647959及CPU监测，仍要求全部场景200×两种子nominal/robust/default CPU和视频正常头颈，不取消难姿势或放宽验收。

正式CPU监测另发现工具故障：原脚本未设置MUJOCO_GL，headless服务器首次4000回放因GLFW/无DISPLAY退出1，watcher2052848已终止，不能当作模型运动失败。已修复显式EGL、选自有GPU渲染、隔离PYTHONPATH，并在工具退出时报告cpu_runtime_failed；11项本地CPU监测/回放回归通过。修复同步v2/v3辅助脚本（不改训练物理/奖励），保留失败cpu-runtime.failed-glfw.log；仅核对并退役已死亡/空闲且无子评估进程的4个CPU watcher，未重启训练。新CPU watcher：v2起身2667810、坐站2667839；v3起身2667841、坐站2667883。4000正式CPU评估进程2667834已真实运行并生成prone逐例JSON/NPZ。恢复记录cpu-watch-egl-recovery.json；启动哈希清单是当时快照，辅助脚本后续修改另记，不冒称当前所有脚本仍等于启动哈希。目标未达标，无实机操作或部署。

## 2026-10-10 第4250轮头颈复测与独立候选

上轮分类为progress：4000两模型CPU全部场景8例×两种子的小样本通过，并实际审阅视频发现折颈。本轮复核两训练及四监测实际PID仍存活，初次计数4211/4157。4000正式GPU nominal正在执行，起身seed1009六场景200/200，坐站seed1009四场景99.5%～100%；这是局部结果，未宣布完整验收。

独立导出4250起身权重，按同一默认CPU runtime、seed1009全部六姿势各1例无重置回放（worker2589308，exit0）。六姿势数值全部通过，但视频0/3/8/14秒帧仍严重折颈，终态最后3秒最大头颈HOME误差2.708～2.709rad，与4000约2.707～2.708rad基本相同。证据checkpoint-4250/getup/cpu-runtime、head-posture.json及pose-review.png。此结果只说明加大惩罚后的早期250轮尚未改善头颈，不证明后续必然无法改善。

准备独立getup-headmax候选：只将head_pose_bias的一秒EMA绝对误差从四关节均值改为最大值，保留原高度/倾角门控及reset逻辑；权重固定当前阶段1.5，其他奖励、动作EMA/观测、原版限位与物理均不变。单关节2.7rad误差现直接计2.7而非0.675，倒地门控关闭时仍零惩罚，避免限制头部辅助恢复。服务器真实Torch验证平均稀释、门控关闭/部分开启、全零有限值以及原任务奖励/动作配置保持通过；候选注册独立任务Mjlab-GetupHeadMax-Smoke-MicroDuck-HD1910，已启动64环境/5轮冒烟，launcher2609586，日志runs/getup-headmax-smoke-20261010.log。尚未长训，不改正在运行的v2任务，不将候选检查当作成功率。

冒烟已完成exit0、5轮、nan_state0，head_pose_bias实际为负（最后日志−0.0014），不是权重为零而跳过的新函数。已下载日志。4000 GPU nominal完整验收两项通过：起身6姿势×2seed各200例全100%，坐站8组99%～100%，scopeComplete=true/accepted=true；独立证据两项gpu-nominal-summary.json。robust仅部分结果：起身seed1009前5姿势100%；坐站seed1009 stand_cycle168/200=84%、sit_cycle176/200=88%、两保持98%/98.5%，因此整体仍未达标。剩余robust继续原监测，不能将部分JSON的accepted=false当作已完成全部测试；后续针对坐站重复曝光和起身终态头部各自独立续训。最后实查两训练4379/4328，仍运行，CPU正式验收尚未启动。

## 2026-10-10 第4000轮检查与起身奖励审计

本次实查训练getup4043/12000、sitstand3959/10000，实际训练PID2052833/2052857及四个监测PID均存活。起身4000检查点已导出，GPU六姿势初筛正在执行；另启动默认CPU runtime全部六姿势、seed1009各1例视频诊断PID2553393，输出checkpoint-4000/cpu-runtime-probe。单例诊断不构成95%验收；没有重启训练或修改活动任务。

随后4000起身CPU单例六姿势数值全部通过，视频逐姿势0/3/8/14秒帧已检查：都能站起，但终态仍严重折颈，最后3秒四头颈关节中最大HOME误差均约2.707～2.708rad。证据pose-review.png、cpu-head-posture-review.json。不能由六次成功推算95%，也不能将身体站起计为头颈姿态正常。GPU当时仅前五姿势64/64，倒置未结束，scopeComplete=false。进一步已启动两模型默认CPU各场景8例×seed1009/2027诊断，实际PID2568096/2568110，均保留原runtime滤波、15ms目标延迟和所有场景，不挑选有利运行配置；正式GPU与CPU验收仍由原监测执行。最新实查训练4106/4049轮，目标未完成。

追加完整4000轮初筛：起身GPU六姿势全部64/64，scopeComplete=true，但单种子64例仍非正式验收，原监测已进入held_out_validation。默认CPU两种子、每场景8例：起身12组全部8/8共96/96，坐站8组全部8/8共64/64，两个scopeComplete=true/accepted=false（未达到200例×两种子样本要求）。证据checkpoint-4000/{getup,sitstand}/cpu-runtime-8probe-summary.json及getup/gpu-probe-summary.json。4000模型相比3000的重复坐站和多姿势恢复数值已有明显进展，但不能省略robust、正式CPU样本及视频头颈质量；继续原训练，不启动替代候选或部署。

3000起身权重在3700轮奖励课程下的64世界审计已完成，证据checkpoint-3000/getup/reward-basins.json。仰躺/左侧失败状态倾角约66°、高度约64mm，仍获得净奖励约+2.76/秒，其中腿HOME跟踪约+2.00、upright_linear约+0.61、head_pose_tracking约+0.52；机身已直立的组颈部最大HOME误差约2.715rad，仍获约+10.67/秒。源码显示头部跟踪是四关节Gaussian平均，一个关节极端折叠仍保留其他关节得分；standing_composite仅要求腿部姿态，未要求头颈正常。当前审计表明奖励存在失败状态得分和折颈终态漏洞，但不证明这是所有失败的唯一原因。4000开始头部偏置权重由0.5升至1.5，先检查新权重的实际视频和场景结果，再决定独立奖励候选；不能把课程开启等同于效果改善。

## 2026-10-10 v2第1000轮评估与CPU合同纠错（尚未达标）

**3000轮robust完整验收未通过：** seed1009 stand_cycle165/200=82.5%、sit_cycle166/200=83%、stand_hold199/200、sit_hold198/200；seed2027 stand_cycle174/200=87%、sit_cycle172/200=86%、stand_hold197/200、sit_hold195/200。scopeComplete=true/accepted=false，文件gpu-robust-summary.json。nominal通过不能覆盖robust/CPU失败。两种独立单因素CPU诊断（同3000轮模型、每组8次）：仅去gyro滤波四组循环12.5%/87.5%、50%/50%；仅去jointvel滤波37.5%/75%、37.5%/75%，并无统一改善。保持default runtime滤波系数，不为追分改运行合同；9项CPU回归测试本地通过。

重复切换候选训练方案已实际完成服务器64世界/5轮冒烟：独立注册临时任务Mjlab-SitStandRepeat-Smoke-MicroDuck-HD1910，仅episode_length_s由15改30，4～7秒命令采样、2秒内部目标坡道、奖励值/参数、物理/DR/61→14合同原样保留，增加重置前连续切换曝光。脚本先断言原任务仍15秒及奖励完全一致；5轮exit0、nan_state0，日志sitstand-repeat30-smoke.log。任务和runner仅由runs/train-sitstand-repeat-smoke.py注册，未写活动源码/替换两项训练，5轮只是可训练性检查，不是成功率或已启动正式改进训练。候选后续需要与4000轮表现及同种子CPU循环比较，再决定从合适检查点独立续训。已确认原两训练及四监测仍存活，最新3644/3542轮。

**第3000轮最新：坐站GPU nominal独立验收已通过，但整体目标未完成。** seed1009 stand_cycle194/200=97%、sit_cycle195/200=97.5%、两保持200/200；seed2027 stand_cycle192/200=96%、sit_cycle198/200=99%、stand_hold198/200=99%、sit_hold200/200。全部场景各≥95%、scopeComplete=true/accepted=true，文件checkpoint-3000/sitstand/gpu-nominal-summary.json。robust实际评估PID2465811仍在执行，尚未拿不完整结果宣称通过；CPU与getup亦未通过。

IMU纠正后的3000轮CPU8例×2seed失败门槛已逐端点分析：所有站起端点倾角/高度/腿姿态/速度全部8例通过；stand_cycle的第二次坐下高度/腿姿态分别失败5/8和6/8，sit_cycle的坐姿端点失败1/8和4/8，部分还在平移/转动。此阶段主要是重复下降和坐姿收敛，而非继续沿用旧IMU错误时“站不起”的结论。证据cpu-gate-diagnosis.json。相同种子/初态抽样的诊断：runtime增加1帧jointvel滞后，循环成功率分别50%/87.5%、75%/37.5%，改善和退化并存仍未达标；direct去滤波并缩短目标延迟15→10ms，四组循环全0/8。direct同时改多个因素，不能作为纯延迟AB因果证明；非生产配置/仅两个场景/8例，全部accepted=false。CPU默认runtime滤波、15ms目标队列、无额外观测滞后继续保留；不挑诊断高分取代原95%验收。

**最新CPU根因纠错：角速度坐标系错误已复现并修复。** `mj_objectVelocity(BODY, local=1)`返回惯性主轴坐标，不是训练`root_link_ang_vel_b`使用的机身坐标；CAD的body_iquat非identity，最小模型机身[1,2,3]会读成[2,-1,3]。同状态61维观测审计中，旧CPUgyro与训练最大差2.313rad/s，而关节/动作/命令均约1e-7或0；姿态还有mj_step后xmat未更新的问题。CPU现用freejoint机身角qvel，并先mj_kinematics刷新姿态，不重算BAM使用的上一步约束载荷。修正后同状态clean8秒对照61维全部块最大误差：gyro8.29e-7、gravity3.12e-8、jointpos8.08e-8、jointvel2.96e-7、prevaction/command0。8项本地测试通过，含非identity惯性轴和积分后姿态更新回归。

第2000轮相同seed1009/default runtime/.4秒静置单例，角速度修复前两循环0/1、修复后两循环和两保持全部1/1；视频已确认坐/站互换且头姿正常。证据checkpoint-2000/observation-audit{-corrected}-summary.json与sitstand/cpu-imu-frame-corrected/{summary.json,cycle-review.png,视频/轨迹}。这证明工具错误影响判断，不能把旧回放失败全部归为模型/求解器问题，但一例不是95%验收。此前native物理+GPU观测混合闭环，以及去扰动native-clean/GPU-clean均能站起；此诊断不是完整生产CPU链路，也不代替验收。仅取消静置并补速度1帧延迟时两循环仍失败，未通过削弱启动条件绕开问题。

第3000轮完整GPU64例初筛：起身俯卧64/64、仰躺0/64、左侧0/64、右侧64/64、斜倒64/64、倒置51/64；坐站四场景64/64，监测进入正式独立验证。修正IMU后的CPU起身单例仍仅俯卧/斜倒数值通过，其余四失败，视频俯卧/斜倒仍折颈，不计完整动作通过。坐站CPU两种子各8例初筛：seed1009 stand_cycle3/8、sit_cycle7/8，seed2027 stand_cycle2/8、sit_cycle4/8，两种保持均8/8；scopeComplete=true/accepted=false。因此修正IMU不等于已解决所有泛化问题，下一步对循环失败端点/启动扰动/控制延迟做证据对照。模型和任务继续训练，未部署；最新实查起身3218/12000、坐站3185/10000，两训练及四监测存活。

**本次最新覆盖：2000轮坐站完整nominal验收未通过。** seed1009：stand_cycle192/200=96%、sit_cycle181/200=90.5%，两个保持200/200；seed2027：stand_cycle193/200=96.5%、sit_cycle190/200=95%，两个保持200/200。scopeComplete=true/accepted=false；不能把64例初筛每场景≥95%当作正式通过。robust验证还在执行，两实际训练仍存活。正式结果本地checkpoint-2000/sitstand/gpu-nominal-summary.json。

后续robust完整结果亦未通过：seed1009两循环176/200=88%、174/200=87%；seed2027两循环177/200=88.5%、168/200=84%；四组保持均200/200。scopeComplete=true/accepted=false，随机扰动下切换明显弱于保持。已下载gpu-robust-summary.json，两种子/两profile证据完整，目标继续未达标。最新核实训练2456/2399，两训练及四监测存活；不因评估进程正常退出误判训练停止。

CPU/GPU实际组装参数对照发现另一个检查盲点：v2虽选择68个body_ground网格和两脚，实际编译后仍保留两个**未命名**leg/leg_2自碰撞网格2/2，活动几何总数73（含terrain）。之前“完全仅地面接触”描述不准确，继承默认类造成未命名网格无法仅靠名称列表证明关闭；power_support在训练中关闭，而原CPU仍2/2。CPU一度把三个旧网格全关闭（71活动几何），这一中间版本保存在cpu-contact-corrected，仅诊断且不能作为完全匹配训练的回放。随后恢复两个leg2/2、关闭power_support，保留电机继承ctrlrange±10/ctrllimited=true，按实际v2编译配置复测cpu-actual-masks：两种循环仍0/1、两保持1/1。因此不能宣称自碰撞差异是切换失败唯一原因，更不能从本次修正宣称CPU成功率达标。7项本地CPU配置/时序测试通过。

参数证据checkpoint-2000/cpu-model-audit.json，逐项核对命名body/关节/几何/actuator；活动未命名网格另列mesh/body。命名部分只有ctrl限制开关差异（已修正），没有质量/惯量/关节范围差异。该审计未证明观测、初始状态、全部未命名网格及CPU/Warp动力学逐步一致；后续需相同初态和目标序列的对照。未改变正在训练的模型、碰撞、奖励或课程，也未停止/重启任务；218项启动哈希清单仍仅表示启动时快照。

相同初态/扭矩诊断进一步发现：nominal/play配置重置后仍随机化物理字段，`env.sim.mj_model`默认值不等于Warp当前world字段。单世界seed1009的质量最大差7.10g、惯量5.26e-5、重心2.40mm、dof_armature1.75e-4、geom_friction0.358；此前只比较编译模型默认参数的结论不能证明动态参数一致。诊断现将这些world字段复制到同一MJB的native模型，GPU每个物理步的电机扭矩/干摩擦/粘滞阻尼作为oracle输入回放，绕过CPU观测与控制延迟。同步Feetech内部目标状态及kd_scale后，相同状态下NumPy/Torch BAM扭矩最大误差5.96e-8Nm，未发现公式数值差异。

复制world字段后第1物理步qpos最大差从旧对照8.89e-4降到1.85e-5，第4步1.30e-4；接触后第10步约4.40e-3、第100步7.80e-2，长期轨迹仍分叉。qpos最大误差混合根位置/四元数/关节单位，不能全称为弧度。这是单世界开环oracle扭矩实验，不是CPU闭环成功率，也尚未排除初始solver缓存/其他动态字段或接触数值实现差异，不能直接宣布原生与Warp求解器是唯一原因。证据physics-replay-worldfields-summary.json；早版physics-replay-summary.json未复制world字段且首步内部目标状态未同步，仅留历史诊断。继续训练不修改活动配置，后续针对接触与同初态闭环追踪，不以放宽GPU/CPU验收掩盖分叉。

追加逐物理步重置到GPU状态的诊断（physics-reseed-summary.json）：每步复制qpos/qvel/qacc_warmstart/外力，用当前GPU扭矩、摩擦和阻尼预测下一步，再与真实GPU下一步比较，避免长期误差累积。1599次单步残差qpos最大误差中位1.42e-4、最大1.90e-3；qvel最大误差中位0.0291、最大0.3797；相同状态BAM扭矩最大5.96e-8Nm。没有发现控制公式的巨大误差，长期开环分叉不能据此直接归因唯一求解器缺陷。仅一世界8秒oracle实验，未覆盖CPU自有观测滤波/延迟、.4秒静置、全部场景或正式成功率。startup随机化在env构造时发生，reset(seed1009)并不固定所有构造前随机样本，两次实验world参数不同；每次均从该实验GPU世界复制参数，但不得把跨实验差异当严格同随机参数AB效果。下一阶段需固定构造seed、同初态CPU自有闭环及3000轮评估。

第1000轮nominal初筛每场景64例、seed31415：起身俯卧64/64、仰躺0/64、左侧0/64、右侧60/64、斜倒64/64、倒置0/64；坐站stand_cycle44/64、sit_cycle48/64、站姿与坐姿保持均64/64。起身失败主要是机身倾角和高度，不能用腿关节接近HOME代替身体站起；坐站失败主要在坐姿端点的高度/腿姿态及部分平移。初筛不满足200例×两种子/nominal与robust完整验收，且视频头部姿态尚不正常，目标未完成。

CPU工具此前硬编码500Hz/Euler且禁用Euler damping，与实际训练200Hz/implicitfast、decimation4、solver30/line-search50不一致；BAM CPU controller也没有自动实现JSON中的command_delay。已修正元数据记录真实模拟设置，CPU依据合同显式排队识别延迟6.005ms，加runtime传输8ms；5ms步长向上量化后runtime15ms/direct10ms。旧500Hz证据保存在cpu-diagnosis/cpu-direct，不再用作训练动力学一致性的证明。checkpoint-1000元数据已刷新并保留旧副本，模型权重与活动训练物理/奖励没有更改。

正确合同下seed1009单例仍有多姿势起身和两种坐站循环失败，不能把问题归因于步长并宣布解决。新增仅诊断的--joint-velocity-lag-steps选项，核对训练joint_vel固定1控制帧滞后；非零选项不能计作runtime正式验收。相同1000轮坐站模型、runtime/lag1两种循环仍0/1，说明仅补这一项时序差异不足以解决失败，后续需更完整的初态/观测/目标与BAM动力学对照。6项CPU判定/时序合同测试本地通过；服务器venv无pytest，未冒称服务器pytest通过。服务器诊断输出checkpoint-1000/cpu-velocity-lag1。

本次实查getup1928/12000、sitstand1973/10000，两实际训练PID2052833/2052857和四个GPU/CPU监测进程均存活。下一检查点2000由现有监测继续导出与评估；不重启训练、不修改活动奖励、不连接主板。实时轮数以runs/report-skills-v2.py的进程核验结果为准。

第2000轮CPU runtime/seed1009每场景单例并视频复核：起身俯卧与斜倒数值通过、其余四姿势失败；俯卧末帧仍严重折颈，不能计完整姿态通过。坐站两种保持通过，两种循环失败；站→坐端点稳定，但坐→站阶段高度仅约51～58mm（站立目标115mm），倾角约21～34°，腿姿态误差约0.57～0.84rad。相比1000轮已有下降阶段/静态站立改善，仍缺起身过渡。视频拼图及无重置轨迹位于artifacts/hd1910-skills-20261009/checkpoint-2000/{getup,sitstand}/cpu-early-runtime；单例仅诊断，accepted=false。两项2000轮GPU64例初筛进程已核实运行，结果尚未完整，不能填报完整GPU成功率。

随后2000轮坐站GPU初筛完整：stand_cycle61/64=95.3125%、sit_cycle63/64=98.4375%，两种保持64/64；已达到初筛门槛，监测将执行两种子各200例的nominal与robust正式验证。该GPU进展与CPU单例起身失败并存，不将GPU初筛改写为最终验收。最新训练2090/2129，两训练及四监测存活；起身2000轮初筛目前仅前四场景完成（俯卧64/64、仰躺0/64、左侧0/64、右侧46/64），scopeComplete=false，其余不得填入推测结果。

## 2026-10-09 HD1910多姿势起身与双向坐站重新训练（进行中）

**最新覆盖：v1物理环境错误已定位，已停止自有两项训练，保留日志/权重，v2重新训练。** 原先只编译`spec_fn()`的检查遗漏了Entity组装：继承的`CollisionCfg('*_collision', disable_other_geoms=True)`把全部68个`body_ground_*`机身网格接触关闭为0/0。实际GPU坐姿保持33/33倒地，身体可穿过地面，CPU回放则保留接触；此前GPU和CPU物理合同并不一致，不能把v1训练作为本机完整接触环境的成功训练。初始EMA对齐/去除扰动仍0/33，实际第一帧目标与坐姿逐项误差0，组装后的二进制模型证明68个掩码均为0，证据`skills-gpu-pose-hold-details.json`和本地`nominal-clean-compiled.mjb`。

修复：技能环境独立碰撞选择包含两脚代理与全部68机身网格，contype1/conaffinity0/condim3，脚priority1，保持机身仅与地面接触、不增加视觉网格自碰撞。回归测试先应用Entity collision edits再编译，避免旧检查的盲点；28项配置/重置/14变体限位检查通过。GPU实际组装掩码恢复68项1/0，无扰动保持站31/31、坐33/33；原始±5°/关节扰动下，nominal和training两种配置站30/31、坐33/33，坐姿高度约61.3mm，P95倾角约6.2°。这些为5秒静态目标保持诊断、非模型成功率。CPU回放同步碰撞掩码及BAM额外惯量；导出元数据应用碰撞配置、检查68实际接触并记录v2物理版本/技能配置SHA256，旧版本不能通过CPU正式accepted。

v2隔离目录`/home/leiwu30/microduck-training/microduck-rl-skills-v2-20261009`，218个源码/几何/舵机参数/相关工具文件逐项SHA256与本地清单匹配。两个64环境5轮冒烟exit0、nan_state0，才启动4096环境正式训练：`hd1910-getup-v2a-12000` GPU4，launcher2052831/train2052833/GPU监测2052835/CPU监测2052848；`hd1910-sitstand-v2a-10000` GPU6，launcher2052854/train2052857/GPU监测2052862/CPU监测2052871。保持原多姿势覆盖及95%独立场景验收，未加载v1错误环境权重。已核实实际训练和监测进程存活，启动不代表正式成功率达标。证据`source-manifest-v2.json`、`skills-v2-launch.json`、`skills-v1-invalid-physics-stop.json`、`skills-v2-gpu-pose-hold*.json`。只停止核验cmdline匹配的1811223/1811238，未动其他GPU作业/主控/EEPROM；旧3000以后课程是否合适仍依据新检查点评估。

v2导出链路验证：两个5轮冒烟检查点均成功导出61→14 ONNX及元数据，元数据检查Entity碰撞编辑后的68个有效机身接触；服务器原生CPU runtime回放六起身+四坐站场景各1例有限值/无重置完成，scopeComplete但全部0%成功且accepted=false，仅验证工具链，不是训练结果。服务器`exports/skills-20261009/v2-smoke/`存模型/元数据/contract日志/CPU轨迹和verification.json，本地`artifacts/hd1910-skills-20261009/v2-smoke/`存验证文件及模型。GPU验收额外逐次检查实际组装模型的68项1/0/condim3及terrain接受bit1，错误物理合同直接拒绝；8项判定/覆盖/掩码检查通过。该评估助手改动不改正在训练的物理/奖励；218项启动清单保留历史含义。正式训练最新实测进度167/176轮，两个训练及四监测进程存活、nan_state0，95%正式验收仍未证明。

v2第250轮早期检查点：独立nominal/seed31415/每场景64次无重置ONNX初筛完成。起身俯卧64/64、斜倒64/64，仰躺/左右侧倒/倒置各0/64；坐站坐姿保持64/64，站姿保持20/64，两种反复切换各0/64。全部有限值、scopeComplete，但不足正式200次×2seed且场景未齐达标，accepted=false。模型及结果在`exports/skills-20261009/<v2 run>/early-250/`，本地`artifacts/hd1910-skills-20261009/early-250/`。

本地CPU runtime单例seed1009/视频复核：俯卧身体撑起、数值稳定判定通过但头明显低垂（末帧头关节HOME最大误差2.712rad）；仰躺停在地面；坐姿保持稳定；站姿起始循环能下降，重新站起阶段倒下。不能把俯卧数值100%标为完整动作验收，头部及视频质量仍需改善/核验。四个单例仅诊断，非成功率统计；见`early-250/video-review.json`及对应`cpu-diagnosis/`的MP4/帧图/轨迹。正式训练继续，最新进度536/557轮，两训练与四监测进程存活。尚在早期课程，不为改善表格而放宽标准或缩减倒地姿势。

以下v1历史检查点与启动信息保留追溯，已由上述物理修复覆盖，不能继续当作当前运行状态：

最新检查点证据：训练实际进程1811223/1811238仍运行，已到1226/1271轮。1000轮独立64环境初筛：起身六种倒地姿势各0/64；坐站的站姿保持64/64，坐姿保持、从站/坐开始的重复切换各0/64，均未达标。导出的同一权重本地CPU runtime单例视频复核：仰躺仍躺着、站姿循环忽略坐下、坐姿保持塌下去；单例仅诊断，不代表成功率。证据`artifacts/hd1910-skills-20261009/skills-progress-20261009.json`、`getup-1000-diagnosis/`及`sitstand-1000-diagnosis/`。当前保留训练，不依据早期检查点直接改奖励或重启。

坐站采样诊断：新增`scripts/server/diagnose_skill_commands.py`，在同一任务训练配置下独立64环境/seed8191/30秒运行1000轮ONNX，未发生终止或重置。actor指令槽48:51与命令管理器逐帧最大误差0，59/64环境见过两种姿态旗标，旗标实际切换0～6次，采样时刻坐下比例约47～55%，内部alpha随命令变化。其余5个环境在有限随机窗口只见一种旗标，不能据此认定采样器损坏。平均高度约50～51mm且视频所见失败仍在；该结果排除这次诊断里的命令槽未写入/训练采样全零，不能证明所有训练时刻或坐站已经成功。证据`sitstand-1000-command-audit.json`。继续等待后续检查点的场景表现，不以reward增长单独判断动作达标。

CPU监测故障已修复：`checkpoint-*`误匹配同前缀probe日志文件，产生NotADirectoryError并退出。新增只选`checkpoint-数字`目录及回归检查；核实旧PID不存在后归档PID文件、恢复两个CPU监测，新PID1936901/1936905且已核实存活，未重启训练/GPU监测。评估补充逐稳定条件失败诊断和完整场景×种子集合检查，拒绝重复场景替代缺失场景；服务器6项相关检查通过。以上改动仅评估/监测工具，不改正在训练的奖励/物理配置。95%验收仍未满足，目标保持进行中。

用户目标：在更新后的本机STL/HD1910参数上重新训练两个模型，成功率各≥95%，倒地起身覆盖多种姿势。源代码隔离到服务器 `/home/leiwu30/microduck-training/microduck-rl-skills-20261009`，复用原项目已锁定依赖的`.venv`，通过PYTHONPATH指定隔离源码，不改其他人的GPU作业。

任务：`Mjlab-Getup-Flat-MicroDuck-HD1910`、`Mjlab-SitStand-Flat-MicroDuck-HD1910`。使用replica更新STL、1910_m6.json、XgoDuck14关节物理限位、BAM kp5/7.4–8V/电压下降随机化、EMA旧0.45及原始上一动作，保留61→14合同，不额外裁模型目标。从本机robot_walk生成68个仅与地面接触的机身网格，避免视觉网格相互重叠产生虚假自碰撞。保留质量/惯性、摩擦、编码器偏差、IMU误差和延迟随机化；不加训练外的硬件运动。

训练前CPU姿势保持：16次带关节扰动站姿3秒，末1秒倾角P95最大7.17°、平均躯干高0.11505m；坐姿16次P95最大3.64°、高0.06240m，均有限值。目标高度据此设置，不沿用旧几何高度。证据本地`artifacts/hd1910-skills-20261009/pose-hold.json`。20项配置/限位检查通过；两个任务64环境5迭代均exit0/nan_state0，归一化ONNX61→14及有限值核验通过。独立8环境验收脚本冒烟通过（未训练的短模型0%成功，符合预期，不能作为达标模型）。

首轮正式训练：`hd1910-getup-v1a-12000`，GPU4/4096环境/12000轮，launcher PID1811220、train PID1811223、watch PID1811253；`hd1910-sitstand-v1a-10000`，GPU6/4096环境/10000轮，launcher PID1811233、train PID1811238、watch PID1811265。默认训练seed42；PID只是启动时识别，后续必须核对进程与退出文件。首次无a的两个run因误用`--seed`（应为`--agent.seed`）在参数解析时exit2，未训练，保留失败日志；修正版移除多余参数，已核对训练迭代日志增长。

起身训练按课程保留站姿/坐姿引导，逐渐到95%倒地起始：俯卧、仰躺、左侧、右侧、斜倒、倒置，各带随机yaw、±12°姿态扰动和关节扰动；没有将仰躺/俯卧的平均成功率代表全部姿势。坐站使用50%站/50%坐初始姿态、0站/1坐的命令标志，4–7秒随机保持/切换，保留2秒目标斜坡。

`watch_skills.py`每约1000轮导出并做64环境全场景初筛；候选通过或训练结束时，用独立seed1009/2027、每场景每种子200次，在nominal和robust两种配置下验收。起身每姿势15秒、末3秒至少95%采样倾角<20°、目标高度误差<12mm、水平速度<0.2m/s、角速度<2rad/s、腿平均目标误差<0.25rad。坐站验收从站/坐开始的重复双向切换及两种静态保持，每个目标末2秒满足相同稳定条件，且全过程倾角<45°。每一场景/种子成功率≥95%，任何终止或重置算失败，64环境初筛不能标accepted。通过后自动生成全场景视频，仍需看视频及CPU部署回放，成功率为所述仿真试验结果，非实物成功率。

追加核验：28项检查通过（新增8种姿势的部分环境重置、关节范围、四元数归一化与地形高度偏移检查），服务器`runs/skills-config-tests-v1.log`保存结果；306个源码/几何/参数/测试文件SHA256与本地清单全部一致。训练日志首次核对已到137/140轮，训练及两个watch进程存活，不再以启动器PID或旧状态文件单独判断任务运行。

CPU部署方式验收补充：新增`scripts/evaluate_skills_cpu.py`，不依赖mjlab/GPU，用元数据中的HOME/坐姿/实测高度及六倒地姿态，验证ONNX和63个replica XML/STL文件及舵机参数SHA256。按500Hz物理/50Hz推理、raw prev_action/EMA旧0.45、平台近似观测滤波（joint velocity旧0.16/gyro旧0.25）和额外8ms目标队列，加BAM识别延迟，无自动重置或模型切换；记录全轨迹、接触数、NPZ及视频。判定与GPU相同，同时要求稳定窗口有完整150/100帧，不让提前截断或局部样本通过。

4项CPU判定测试通过：不完整/非有限轨迹拒绝、95%稳定窗口边界、坐站每个端点及中途倾倒检查、多角度四元数。5轮短模型在本地六起身+四坐站场景有限值/无重置冒烟通过，均0%成功，不是正式模型验收；服务器原生CPU俯卧冒烟也通过。证据`artifacts/hd1910-skills-20261009/cpu-rehearsal-verification.json`。已启动独立CPU监测：getup PID1894715、sitstand PID1894725；仅当相同ONNX的nominal和robust GPU独立验收均accepted时，自动每场景/种子200次runtime CPU回放及视频，所有场景各≥95%才标CPU accepted，最终仍须看视频。CPU监测及元数据扩展不改变已运行的训练任务/奖励/STL/参数；原始训练启动清单保留不改写。没有主板连接或部署。

本目标尚未完成：正式模型的95%成功率、视频验收和部署方式CPU回放均未验证；当前仅启动首轮训练，不承诺加轮数必然达标。输出在服务器`exports/skills-20261009/<run>/`，训练/监测日志和状态在`runs/`。无主控连接/机器人部署。后续以检查点表现诊断课程或奖励再续训，不以任务启动或高reward代替验收。

2026-10-01：训练工作目录迁到服务器，正式训练在服务器执行。本地保留代码维护与文档。SSH 密码不存入仓库。

## 环境与目录

- SSH：`leiwu30@8.133.210.70`，端口 `8422`。
- 系统：Ubuntu 24.04 x86_64，8张 NVIDIA RTX 4090；显存以 `nvidia-smi` 为准。
- 项目：`/home/leiwu30/microduck-training/microduck-rl`。
- 环境：项目独立 `.venv`，Python 3.12，`uv sync --locked`。
- 源码基线：官方 `pollen-robotics/microduck_rl`，`develop`，`cfe1c2adcceb55f6b6e369c888b31c6873175c55`。服务器由本地 Git bundle 克隆，保留历史。
- 启动日志、PID和完成状态：`/home/leiwu30/microduck-training/runs/`。
- 训练断点、TensorBoard指标和配置：项目 `logs/rsl_rl/<experiment_name>/`。
- ONNX输出：`/home/leiwu30/microduck-training/exports/`。

本机使用专用 SSH 密钥 `C:/Users/wl/.ssh/microduck_training_ed25519`，独立主机记录 `C:/Users/wl/.ssh/microduck_training_known_hosts`；私钥与密码均不上传到训练目录。

```powershell
ssh -p 8422 -i C:/Users/wl/.ssh/microduck_training_ed25519 -o UserKnownHostsFile=C:/Users/wl/.ssh/microduck_training_known_hosts leiwu30@8.133.210.70
```

## 短测试与正式训练

脚本默认仅使用物理GPU7；登录后先检查占用，必要时通过 `MICRODUCK_GPU` 选择其他空闲卡。`CUDA_VISIBLE_DEVICES` 映射后，程序中的 `cuda:0` 对应所选物理卡。默认 TensorBoard 本地日志，无需 W&B 登录。

```bash
cd ~/microduck-training/microduck-rl
nvidia-smi

# 每个新配置先64环境、5迭代；这不是有效训练成品。
MICRODUCK_GPU=7 scripts/server/smoke.sh Mjlab-StandUp-Flat-MicroDuck

# 正式训练示例，按后续任务决定是否启动。
MICRODUCK_GPU=7 scripts/server/launch.sh standup-001 Mjlab-StandUp-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max-iterations 1000 --agent.run-name standup-001

tail -f ../runs/standup-001.log
cat ../runs/standup-001.exit  # 完成后出现，0为正常退出
```

断点续训与跨任务暖启动不同，后者须遵守 AGENTS.md 的课程与归一化要求。

```bash
MICRODUCK_GPU=7 scripts/server/launch.sh standup-resume-001 Mjlab-StandUp-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max-iterations 1500 \
  --agent.resume True --agent.load-checkpoint model_999.pt
```

以上续训按上游最新运行目录选择机制查找断点；执行前核对帮助与所选运行，避免误接其他任务的断点。

## 导出

```bash
scripts/server/export.sh Mjlab-StandUp-Flat-MicroDuck \
  /absolute/path/to/model_999.pt ../exports/standup-001.onnx
~/.local/bin/uv run --locked python scripts/server/verify_onnx.py ../exports/standup-001.onnx
```

使用上游导出器，包含观测归一化；导出后再检查输入61维/输出14维和有限数值，并做仿真回放。短测试模型不能直接用于实机。

## 验证边界

官方仿真执行器仍是XL330 BAM模型，尚未适配本机HD1910。环境安装、短训练和导出成功仅证明服务器训练链路可用，不证明站立、起身或实机迁移成功。不连接主控、不上传策略到机器人、不自动启动长期训练。

## 2026-10-01 部署验收

- 锁定依赖安装完成，Python3.12、PyTorch2.9.1+cu128，CUDA可用。
- GPU7运行StandUp任务64环境、5次迭代正常结束，nan_state计数均0，保存model_0.pt与model_4.pt。
- 上游导出器成功导出setup-smoke.onnx，CPU ONNX Runtime核验输入[1,61]、输出[1,14]，零观测和随机小量观测输出均为有限值。
- 后台启动器通过帮助命令验收，PID/日志/退出状态文件生成，退出状态0；未启动长期训练。
- 服务器setup/deployment-verification.json保存源码基线、脚本哈希、导出哈希与检查结果。

## 2026-10-05 正常步伐 v11（新训练，尚未验收）

用户反馈旧行走模型低头、小碎步，要求正常步伐。服务器v10审核显示颈部平均HOME误差约31–39度，最终accepted=false；不能把旧模型描述为已验收。新任务 `Mjlab-WalkNatural-Flat-MicroDuck-HD1910` 使用独立实验目录、从头训练4096环境/4000轮，不续用低头策略。正奖励按0.5秒头姿偏差滤波乘以质量因子，取消低头仍支付40%的奖励下限；步幅奖励分阶段增强，保留完整摆腿/离地高度目标。训练动作EMA旧0.45+新0.55、raw prev_action，与当前运行端匹配；本机几何/HOME及61维观测保持。

正式run：`hd1910-walk-v11-natural4000`，GPU7。64环境/5轮短测试与ONNX 61→14有限值核验先行，相关15项测试通过（使用python -m pytest）。训练完成由服务器finalizer自动导出、评估7种指令并生成前进视频；验收包括前进同脚落点步幅至少75mm、离地至少10mm、腾空至少100ms、平均头关节偏差不超过8度及32环境无跌倒。此处步幅是同脚相邻落点距离，不是单步长度。所有结果均为仿真；未更换平台模型、未发实物运动指令。实际正常步伐仍须回放及实机验收。

### 2026-10-05 用户缩短为1000轮阶段评估（覆盖上述4000轮计划）

用户要求先1000轮看效果，再决定是否继续。终止4000轮训练及对应后处理，使用已保存model_250.pt完整恢复权重、归一化、优化器和课程步数；prepare_resume_checkpoint将下轮设251，余749次更新，总计1000轮，最终model_999.pt。新run=hd1910-walk-v11-natural1000，自动后处理目标同步改1000。未保存的250之后少量更新不保留。1000轮后只导出评估和视频，不自动续训；课程尚未到1500轮最终阶段属于本次阶段评估的边界。

### 2026-10-05 v11 1000轮完成：仿真未通过，未续训

训练exit0，model_999完成，导出及7指令评估完成，finalize=needs_more_training/accepted=false。前进32环境20秒无跌倒，但命令0.2m/s时实际平均约0.000062m/s、总前进2.8mm、有效落脚0/步幅0，几乎原地站立；头颈平均HOME偏差最大0.75°、P95最大6.96°。转右32/32环境出现失败。头姿改善不能冒称正常行走成功。模型和视频下载至microduck-rl/artifacts/hd1910-walk-v11-natural1000。按用户要求停在1000轮，未自动续训/未部署实机。当前奖励可能仍使静止更有利，为推测，须后续诊断再改训练。

### 2026-10-05 按用户要求提高真正行走得分（v12配置，未正式续训）

用户要求把得分加到行走。速度跟踪权重5→10、腾空0.8→2，新增command_progress权重8：按指令方向实际速度/目标速度给0–1分，反向与静止为0，超速不额外支付。移动指令下原有正向姿态/头姿/速度等奖励同步乘真实移动进度，消除站着领取大额姿态分；零指令保持原站立奖励。步幅奖励增加交替落脚要求，双脚同时落地和同脚连续跳不领取该奖励。维持头姿和EMA规则。新配置不改变已导出v11权重；仅短测试，不自动开启下一轮正式训练。

### 2026-10-05 用户授权v12正式1000轮

用户明确“训练1000轮看看”。使用行走得分修订配置，GPU7/4096环境/1000轮，从头学习避免直接沿用v11原地站立策略；run=hd1910-walk-v12-progress1000。16项测试与64环境5轮短训练已通过，正式启动前检查新短测试ONNX 61→14有限值。完成自动导出model_999、评估7种指令及生成前进视频，随后停止，不自动续训。未替换平台模型、未执行实物动作；效果以本轮评估和视频为准。

### 2026-10-05 v12 1000轮结果：开始迈步，但仍碎步/走偏，未通过

训练exit0，自动导出/7指令评估完成，accepted=false。前进32环境20秒：实际移动段平均0.236m/s（命令0.2）、前进投影2.82m；1/32环境失败。头关节平均HOME误差最大3.43°、P95最大8.14°。同脚相邻落点步幅中位36.8mm，目标75mm；腾空80ms，目标100ms；离地峰值约8mm，目标10mm。航向偏差P95约57°、横向偏移P95约1.10m。已经从v11原地站立变为移动，但未达到用户正常步伐要求。后退32/32失败，转左8/32、转右10/32失败；指标含自动重置后的片段，失败场景速度/路径不应视为连续成功。模型及视频保存在artifacts/hd1910-walk-v12-progress1000；1000轮后停止，未续训或部署实机。下一轮应加强完整步幅与直线航向约束，先诊断再改参数。

### 2026-10-05 v12 0.05m/s慢速回放（仅仿真）

用户要求看0.05m/s，evaluate_walk新增--forward-speed参数，未更改模型权重/训练。32环境20秒，前2秒站立、2–15秒前进、末5秒停止；移动统计4–15秒。实际平均前进速度约0.031m/s；5/32环境跌倒。同脚步幅中位14.1mm、离地峰值3.49mm、腾空70ms；头颈平均HOME偏差最大10.55°、P95最大14.81°，航向P95偏48.1°。未达到稳定正常慢走，不能把命令速度称为实际速度。统计含失败后重置片段，回放显示第0环境，不代表所有环境。视频及JSON保存在artifacts/hd1910-walk-v12-progress1000/hd1910-walk-v12-progress1000-forward005.*。未续训、未部署或执行实物动作。

### 2026-10-05 v13针对0.1m/s正常步伐（用户“继续”授权1000轮）

新增WalkSlow独立任务/实验目录，不覆盖v12。95%移动样本为正向直行，其中80%精确0.1m/s、其余0.07–0.13；保留约25%零指令起停。v12会迈步模型暖启动，MICRODUCK_WARM_START=1重置新训练轮数/课程计数，保留权重、优化器、归一化；来源自然步态相同61维合同。完整摆腿奖励乘腾空/离地/摆腿距离完成比例，保留交替落脚，目标周期0.6秒/腾空0.16秒/离地15mm，奖励1000轮内逐级增强。直行正奖励增加角速度/侧向速度质量门控，新路径成本锚定每个直行段开始的航向/横向位置，目标跟踪std0.04；进度奖励8→4，减少只靠速度获得高分。

17项检查及64环境5轮短训练通过。计划run=hd1910-walk-v13-slow1000，GPU7/4096环境/1000次新更新，完毕后停止。专属后处理评估0.1m/s前进和零指令：32环境20秒无失败、实际平均0.08–0.12m/s、速度绝对误差<=0.03、航向P95<=10度/横向P95<=80mm、同脚步幅>=40mm/腾空>=100ms/离地>=10mm及头姿/停步。0.1m/s目标步幅不同于旧0.2m/s的75mm要求，不把其冒称已验证自然步态；仍须视频验收。本轮不验收后退/侧行/转向。不部署实机，不执行实物动作。

### 2026-10-05 v13 1000轮完成：0.1m/s起步失败，未通过

训练exit0，导出及idle/forward评估完成，accepted=false。0.1m/s前进32/32环境均fell_over，首次失败2.54–3.56秒，命令自2秒开始即起步0.54–1.56秒后跌倒；自动重置累计389次fell_over，有效落脚0。平均速度0.063m/s及航向指标包含反复重置，不能视为正常持续前进/走直成功。零指令32/32无跌倒。此轮退化，重奖励约束组合/启动分布或课程可能导致，原因未定，不直接归因于训练轮数不足。视频/模型保存在artifacts/hd1910-walk-v13-slow1000。已停训、未部署实机。后续优先比较v12/v13同0.1指令及中间断点，诊断奖励/起步行为再改训练。

### 2026-10-05 v13失步诊断（未启动新训练）

用户要求找原因。新增diagnose_slow_regression.py及summarize_slow_diagnosis.py，evaluate_walk支持--start-moving/--trace。相同WalkSlow物理、0.1m/s、8环境12秒：v12原权重0/8失败、v13 model_0 0/8失败；v13 model_250/500/750/999均8/8失败、有效完整落脚为0。证实已会迈步策略在本轮训练中退化，250轮时即可发现，原先只有短测试和末次验收不足以发现行为回退。v12此条件实际约0.140m/s、同脚步幅23.4mm，仍非正常0.1m/s步态，不能视为达标。

冻结v12在新奖励下回放，首环境4–7秒瞬时straight_quality平均0.1175/中位0.0147，46%时刻<0.01；实际平均绝对yaw约0.427rad/s、侧速0.0489m/s，说明新增瞬时yaw/侧速门控会大幅削弱已有迈步奖励，也会惩罚可交替抵消的自然摆动。训练0/100/250/300/500/700/750/999轮日志stride_quality显示0.0000，而速度tracking至999轮升到0.3934，episode长度约92步（1.84秒），非正常持续步态。三项完整摆腿完成比例相乘、再叠头姿/支撑/速度/瞬时走直门控，造成已有短步很难拿到步幅信号；这属于已确认的奖励压低现象，具体每项对失稳的因果贡献未做单因素续训验证，不将怀疑冒称全部根因。

排除项：v13 model_999与ONNX逐步推理最大绝对误差1.46e-6；取消2秒站立等待、从t0直接行走仍8/8失败，故不是只因回放启动方式。训练/推理EMA一致，未发现本次导出错误。结论是先退回v12会迈步权重，取消瞬时摆动“一票否决”，用慢平均/每个直行段累计航向控制；将抬脚、腾空、步长信号分阶段学习，并在每100–250轮定向验收/退化即停，而非直接增加训练轮数。诊断证据在artifacts/v13-regression/diagnosis.json及summary.json；本次仅诊断，没有新正式训练或实机部署。

诊断补充：冻结v12首环境4–7秒，新奖励各项积分合计约-5.26，其中瞬时yaw_error_cost约-4.52，步幅奖励仅+0.032，速度tracking+0.401。这证明现有迈步行为在新评分下净负分，不只是权重看起来偏重。v13最终首环境起步到0.68秒后跌倒的奖励合计约-1.40（含-5失败项）；可以在倒下前先取得移动正分，移动奖励本身没有要求完整交替步态。这两段时长/状态不同，不用其直接比较长期收益或冒称完整因果消融。trace在ended帧的速度/倾角是自动reset后的状态，只用于确认该帧ended和奖励，不当作跌倒前姿态。

### 2026-10-05 v14修复先100轮对照（用户“继续”授权）

新增WalkRepair任务，保留v12模型/归一化，去掉v13瞬时straight_quality门控，瞬时yaw_error权重-3→-0.3/角速度tracking6→1，保留0.5秒yaw bias及较轻的累积路径成本。步幅分数改为距离/抬脚/腾空三个完成比例的平均，避免三重乘积让短步没有学习信号；抬脚/腾空/摆腿目标为12mm/120ms/周期0.4秒，保持交替落脚及头姿/支撑。只步幅项移除瞬时移动门控，允许先抬脚再前进，不恢复原地移动奖励。速度tracking std由0.04放宽0.08、LR5e-5。

为已有步态微调，v12探索std原0.29–1.07rad，seed最大限制0.2rad，actor/critic和归一化保留，旧目标Adam动量清空，新课程/轮数重置。19项检查、64环境5轮短训练及61→14有限值导出通过。GPU7/4096环境先100轮run=hd1910-walk-v14-repair100；GPU6进行匹配v12基线：32环境20秒0.1命令，1/32失败、实际0.128m/s、同脚步幅21.2mm/腾空75ms/抬脚5.6mm。100轮完成后同条件评估，不将训练得分改善冒称正常步态。后续仅在不明显退化时继续到总1000轮，guard_gait_repair.py每100轮检查，失败/步幅/速度/偏航明显退化终止本run。未部署实机/未执行实物动作。

### 2026-10-05 v14 100轮对照后继续至总1000轮

相同32环境20秒/0.1m/s，修复100轮：失败1/32（原v12基线1/32）；同脚步幅21.15→30.40mm；航向P95 33.23→16.83度/横向P95 0.446→0.232m；实际速度0.1283→0.1213m/s，速度误差0.0333→0.0237m/s；头姿平均最大1.22度。尚未通过正常步态：腾空75→65ms、离地5.61→4.72mm、接触抖动比0.17→0.54，抬脚/腾空未改善。100轮不是最终验收。

已从model_99完整续训到总1000轮，source iter99调整至下轮100，common_step_counter2400保留，共900次余更新；run=hd1910-walk-v14-repair1000。正式PID357024、finalizer357041、guard357054（仅当时进程识别）。每100轮GPU6评估32环境/20秒/0.1命令，较100轮参考：失败>2、步幅或抬脚下降>25%、腾空下降>15%、速度误差增>0.025、航向P95增>50%+5度，终止该run并保留断点。guard失败检查只是退化兜底，不代表正常步态验收通过；最终仍用Slow专属条件及视频。100轮模型/视频本地artifacts/hd1910-walk-v14-repair100，未部署实机。

### 2026-10-05 v14守卫400轮停训，v15优化摆腿发现信号

用户继续优化。v14守卫检查200/300/400轮32环境20秒：均0失败，但同脚步幅26.9→25.8→20.2mm，400轮较100轮30.4mm下降超25%，守卫主动SIGTERM停止，训练exit143；当时后处理failed指的是被守卫终止，不是训练崩溃。未完成1000轮，不能宣称完成。将后续finalizer改为明确stopped_for_regression。

进一步读源码：继承feet_air_time要求current_air_time>125ms才开始正奖励，实际腾空65–70ms，短步无法取得腾空奖励。新增WalkSwing：摆腿中连续反馈脚高/腾空时长，两个比例平均加最短板项，只有单脚支撑+air_time<240ms才能给分；仍乘已有实际移动/头姿/支撑门控，避免静止/双脚跳或永久单脚抬起支付。目标脚高10mm/腾空100ms，地面脚间隙成本目标统一10mm，进度权重4→2、步幅权重固定3、路径成本固定100轮阶段，集中优化抬脚和时长。保持0.1m/s采样、EMA、raw prev_action及本机几何。

21项检查/64环境5轮短测试通过。从v14 100轮model_99继续微调（此断点步幅30.4mm优于400轮20.2mm），保留actor/critic/normalizers、探索std上限0.2rad、清旧目标Adam动量；先新100轮run=hd1910-walk-v15-swing100，结束自动0.1评估/视频再判断是否扩大。未部署平台/无实物动作。

### 2026-10-05 v15 100轮无抬脚改善，v16转移奖励占比短对照

v15 32环境20秒0.1m/s无跌倒，但同脚步幅23.3mm（v14种子30.4）、腾空65ms不变、抬脚4.43mm（种子4.72），航向P95 26.2度（种子16.8）。未通过/不直接续1000轮，视频及模型保存artifacts/hd1910-walk-v15-swing100。连续摆腿反馈有信号但本100轮未改善几何质量，不以稳定或奖励非零称优化成功。

新增SwingFocus作为同种子/同物理/同目标的奖励占比对照：连续摆腿权重3→8，速度跟踪10→5，单纯前进2→1，其余Swing配置不变。22项检查及64环境5轮短训练通过。GPU7从同一v14 model_99独立微调100轮，run=hd1910-walk-v16-focus100，自动0.1m/s 32环境/20秒+idle评估及回放，先检查脚高/腾空/步幅再判断是否继续。此次未自动接1000轮，未部署或实物动作。

### 2026-10-05 用户明确扩到2万轮：v16长训练

用户“训练2w轮看看”覆盖短100轮预算。当前v16短100轮已exit0、model_99保存；继续同一训练至总20000轮（余19900次更新），prepare_resume_checkpoint将iter99→下轮100，common_step_counter2400保留；不重置actor/critic/归一化/Adam/课程。run=hd1910-walk-v16-focus20000，GPU7/4096环境；保存每500轮，预计约9–10小时（按现有1.65–1.7秒/轮估算，非保证）。最终model_19999，finalizer按20000验完整断点并导出/0.1m/s及idle评估/视频。

本次按用户长训练意图中间评估为observe-only：GPU6每1000轮32环境20秒0.1命令，记录质量与退化，保留所有中间模型用于比较；不沿用v14短探针的步幅下降即主动停训。训练原有NaN/异常退出机制保持。guard_gait_repair扩展总轮数/评估间隔/observe-only参数，默认旧模式仍保持。训练不是模型已正常行走，未更新平台策略、未实物动作。

### 2026-10-06 v16总20000轮完成：大步形成，但速度/起停失稳，仿真未通过

训练exit0，实际续训耗时08:45:48，总20000轮/model_19999；导出及评估完成accepted=false。0.1命令移动片段平均0.222m/s（超目标），同脚步幅75.1mm、腾空175ms、脚高27.3mm，头关节平均HOME偏差最大0.81度。32/32前进场景失败：许多发生前2秒零指令阶段，其余多在15秒切停后；idle独立场景也是32/32失败。因此不能将其描述为正常0.1m/s行走或仅起步失败，也不能把重置后混合片段速度/步幅视为连续成功。此轮大幅改善摆腿量，但零指令站立/停步和速度控制退化，未验收/未部署。最终模型及视频/两场景JSON下载artifacts/hd1910-walk-v16-focus20000。下一步应查看中间评估选取折中断点，针对0.1速度上界与起停修复，不直接加轮数。本次仅检查结果/取回证据，没有新训练或实物动作。

### 2026-10-07 XgoDuck原版ONNX对照（无训练/无硬件）

原版模型SHA256 5727b909d5ea1a0f9ff63bf4cea9707339dea6a522374c7a120da633605b923f。原版RL commit326d77a1122870bdefa2c36403937502c958e69c；EMA0.45/raw prev_action/head-body零命令，关闭play默认推扰。32环境20秒起停：0.1指令在原版及本机移植均不迈步；0.2原版实际0.1068m/s、0/32失败、步幅56.3mm/腾空250ms/脚高10.2mm；移植实际0.1114m/s、2/32跌倒、步幅72.9mm/腾空285ms/脚高13.0mm。原版也有速度映射误差，移植仍漂移和跌倒；可作为微调种子，不直接上实机，不以改指令乘2称正常0.1m/s验收。原版配置不确定与ONNX训练版本一致，仿真不含完整实机通信/IMU。视频/JSON/来源详见artifacts/xgoduck-compare-20261007/README.md。新增evaluate_xgoduck.py回放入口；未启动训练或替换硬件策略。

### 2026-10-09 用户要求RL和observer限位对齐（本地配置）

microduck/replica共14个robot XML变体的14舵机关节范围统一为XgoDuck快照；左右髋偏航、颈与头俯仰四项有实质差异，其余仅保留供体精度。颈−135～60°、头俯仰±105°、左髋−0.22～0.45rad、右髋−0.45～0.22rad。STL/BAM/HOME和被动关节不变，HD1910 walk动作裁剪自动读取新XML范围。增加独立JSON快照、CAD导出后处理及16项MuJoCo编译/全部关节范围/导出配置校验；修复相关导出JSON原有尾逗号，防止重新生成恢复旧值。四方向各1秒CPU起身冒烟有限值通过，不作为成功率验收。observer后端/API及网页已同范围，增加全部14项快照比较测试；嘴仍0～30°，模型编码饱和规则保持。无训练/实机连接/部署，旧48次评估证据保留当时哈希。

### 2026-10-09 XgoDuck倒地起身在本机几何/HD1910参数上评估

CPU MuJoCo，共48次无重置试验：仰躺/俯卧/左右侧各4次，三组各16次。当前replica物理限位+起身后切walk：6/16达到直立连续1秒、5/16最终站稳；同STL及HD1910 BAM参数、仅评估时对齐XgoDuck14关节限位：14/16达到直立、6/16最终站稳；对齐限位并持续getup：14/16达到直立、9/16最终站稳。最终站稳要求末3秒至少95%采样倾角<20°、高度>HOME85%、水平速度<0.2m/s、角速度<2rad/s。

发现当前RL颈下限仍−90°，平台/XgoDuck为−135°；原限位标准仰躺卡在颈约−90.5°、躯干倾角28.4°，对齐后约2.38秒达到直立，持续getup结束倾角0.23°，切walk则再次倒下。限位差异和切换均影响恢复；本次对齐全部14关节，不将所有改善独归因于颈。持续getup也非全部成功。保持本机STL/BAM参数，不修改训练XML或部署服务；kp5与硬件增益对应及质量惯性仍需实测验证，仿真不是实机成功率。脚本scripts/evaluate_xgoduck_getup_cpu.py，完整方法、哈希、指标及三栏视频见artifacts/xgoduck-getup-replica-20261009/README.md。未训练、未连接硬件。

### 2026-10-07 XgoDuck站立抗推补充

公开运行端没有独立balance模型，walk训练配置有XY速度扰动±0.2m/s，getup用于倒地起身。本轮walk零指令32环境20秒：5/10秒加0.15m/s四方向/第二次反向根速度脉冲，原版0/32失败、最大倾角5.8度，本机移植3/32跌倒；终止阈值不同，不能严格等价比失败率。仅仿真，不含起身状态机；失败后自动重置不是自主起身。详见artifacts/xgoduck-compare-20261007/README.md，未训练或部署。


### 2026-10-10 objective-v2原因证据、修改与验证边界

GPU nominal各转换窗口：站起所有门通过，坐下主要height/legPose失败、tilt基本通过；审阅首世界1/4/7/10/14/19秒视频均保持站立，与500轮轨迹同方向。原任务10腿关节逐项Gaussian平均，两膝未弯仍由其他关节挣分；49.123mm高度误差按米计/权重6仅扣0.2947，高站姿奖金1.5与命令无关。反事实仅三项之和：SIT旧站着2.7295−0.2947+1.5=3.9348，坐着4.0，区分很弱；不能将其当总奖励或断言唯一原因。运动惩罚的独立贡献、曝光不足、增益映射本轮未证明，不改命令/PD。

新腿姿用exp(−mean(error²)/std²)，高度L1除坐站高度差且权重3，移除upright_while_tall；保留upright_linear/composite。action_rate初期−0.05，1500轮−0.1，2500轮−0.2。新三项SIT错误站姿−2.0428、正确坐姿4.0，STAND相反。旧任务默认height error_scale=1保留；新物理v3合同不变。33项测试、64环境5轮冒烟nan_state0/成本符号、归一化ONNX与CPU四场景/GPU单循环链路完成，5轮成功率0，非动作验收。

隔离服务器/home/leiwu30/microduck-training/sit-contact-objective-v2-20261010，GPU6从头4096环境1000轮，run=objective-v2-firstlook1000，launcher653594；不用旧objective优化器/课程计数，到1000自动停止/导出及GPU nominal/robust与CPU runtime两种子诊断/视频，不自动续训/部署。本地artifacts/sit-contact-objective-v2-20261010/{objective-audit.json,launch.json,smoke.log}，launch含配置/奖励源码/脚本哈希。未连接机器人、发动作或替换板上模型。


## 2026-10-10 脚滑策略偏置优化：长时回归不通过

在同一xgoduck_walk原权重基础上，对五个左右镜像的输出层常量偏置进行三代共36候选的无梯度搜索；不是PPO重训练。新增固定低摩擦、2.5秒摩擦突降和±0.5rad/s真实俯仰角速度扰动，保留61维、原normalizer/隐藏层/EMA/硬件参数。搜索6工况3→6通过，独立16个12秒工况12→14，原成功例未丢；但进一步14个30秒工况7→11，丢失正常地面0.2m/s一例（未倒但前向位移不足/侧向偏移加大），候选拒绝。另0.5倍/0.75倍/首代候选分别6/7/9通过，均丢原正常地面0.4m/s成功例，也拒绝。最终selected=null，metadata与verification已标不通过，不部署、不替换原权重。

所有计数仅筛选：现有前向位移阈值未完整限制侧向偏移，不能称通过即正常直线行走。后续应先增加航向/横向漂移及动态步态验收，再进行状态相关策略训练；静态偏置未找到无退化方案。相同初态重置复现误差<1e-8；ONNX仅输出bias改变、100条推理偏置核对最大误差2.87e-7，原权重哈希保持。证据平级microduck-rl/artifacts/walk-slip-repair-20261010/{final-gate,long-holdout,second-screen,integrity}.json。没有实机动作/部署，未证明实机前摔根因。


LowExplore状态更新：64×5smoke正规导出通过，normalizer逐值保持、参数finite、std均值0.0299766rad落在设定范围；progress=train/100。最终闭环结果未完成，不能称合格。


CriticWarm状态更新：64×5smoke正规导出/校验通过，actor全部state（含normalizer）逐值等于seed、finite，5轮只训练critic已确认。正式1024×120已开始，不重复launcher2586030；完整筛选仍未结束。


ArrivalGate最新：64×5正规导出smoke通过、normalizer逐值保持/finite，1024×100已开始。相同保存姿态的门控静态审计：基线正确3.62°的9项subtotal13.2875保持，错误34.77°的subtotal8.7837→1.7275、stillness1.7748→0；仅证明奖励修改数值达到设计，不是新闭环成功或实机验收。证据gated-static-reward-audit.json，launcher2688946，不重复启动。
