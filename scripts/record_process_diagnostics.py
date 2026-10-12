"""Keep failed training, stricter getup probes, and ongoing repair distinct."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
marker='> **2026-10-10 过程优化与分项诊断最终结果（最新覆盖）：**'
description=marker+''' SitStandProcess-v2的1000轮及CPU/GPU评估全部结束，job.exit=0，训练26分42秒。严格服务器同初态名义28/32→26/32、扰动9/16→6/16、低增益12/16→0/16、高增益13/16→13/16；扰动膝多余行程虽少15.2%，成功退化，selected=null，不部署。最终normalizer逐值保持、训练源哈希未改；两个最终ONNX文件哈希不同但全部initializer及3000条输出完全相等。四个各100轮对照Coverage/WeakProcess/Temporal/Sensor都丢成功例，低增益0/0/0/4（基线12/16），不能唯一归因强平滑税；另状态腿姿奖励参考StateReference100轮也为28/32、8/16、0/16、13/16，排除候选，但未证明或排除计时隐状态的全部影响。仅增动作EMA亦退化，不改服务滤波。原起身ec665…六倒姿正常/物理扰动各48/48，自碰撞诊断亦各48/48，原坐姿/当前v3坐姿额外各16/16；末段抖动小，暂不改起身权重，非实物或正式大样本验收。起身坐下专家直接组合循环失败，冻结传感器历史27例却收敛，不能归因独立历史自振。已收集四条件62条成功闭环示范（训练23250帧/独立seed留出35000帧），监督修复三档、五档冻结其它关节的膝输出修复及三档角速度连续门控全部完成且丢成功例，未部署；拟合回放更平滑不代表实机改善。新增Upright仅在奖励参考到位后惩罚大倾角，保持过渡动作，3项配置/代价测试通过，但GPU6被其它任务占用、启动前拒绝，smoke/100轮尚未运行。已建立当前线程每15分钟接续检查，资源空闲后再验证与训练。板仍sit-smooth-v1，未发运动/部署/改增益标定。用户持续优化/合格自动部署授权仍有效；证据平级RL artifacts/process-repair-v2-20261010/。

'''
for path in [root/'AGENTS.md',root/'docs/server-training.md',root/'docs/hd1910-stand-training.md',
 root.parent/'microduck-observer/AGENTS.md',root.parent/'microduck-observer/docs/实测记录/README.md',
 root.parent/'microduck-observer/docs/实测记录/03-舵机实测.md']:
    content=path.read_text(encoding='utf-8')
    if content.startswith(marker):content=content[content.index('\n\n')+2:]
    path.write_text(description+content,encoding='utf-8')
folder=root/'artifacts/process-repair-v2-20261010'
report='''# 过程优化最终诊断及下一组修复

全部为离线训练/仿真，未发实机动作。用户授权合格后直接部署，当前没有合格新权重。

|候选|正常32例|扰动16例|低增益16例|高增益16例|
|---|---:|---:|---:|---:|
|现用85d7基线|28|9|12|13|
|Process-v2最终999|26|6|0|13|
|Coverage100|28|9|0|13|
|WeakProcess100|28|9|0|13|
|Temporal100|28|7|0|13|
|Sensor100|28|9|4|13|
|StateReference100|28|8|0|13|

全部新PPO候选丢失基线成功trial，不能仅看成功总数。v2扰动两膝多余行程从48.097/40.904°降到43.156/32.316°，平均减少15.2%，但端点退化。GPU robust循环仅7/32、5/32（两seed相同），站保持25/32、26/32，坐保持27/32、25/32。训练job.exit=0只是运行结束。normalizer逐值不变、核心训练源哈希未变；policy.onnx d1995f33…和candidate-999.onnx bc6af06a…全部initializer相同，3000条输出严格相同。server-train/train/final-contract-check.json记录最终评估修订哈希。

低增益退化例stand_cycle/9001/0中，基线起身末段倾角平均3.65°，Coverage为35.00°，WeakProcess34.81°，Sensor3.34°。不是单纯0.25rad腿姿阈值边缘误判：新PPO政策进入倾斜稳定姿态，后续坐下亦未达到目标高度。为何优化选择这个闭环解尚未唯一定位，真实HD1910 P/D映射未知，不据此改实机增益。

状态参考对照只改训练奖励：从当前髋/膝姿态投影推导进度，再向二值指令前导0.1；actor观测/指令、EMA/HOME/限位/自碰撞及执行器不变。两个新状态函数/配置测试与过程/门测试共9通过；64环境5轮导出/normalizer检查后1024环境100轮结束。仍低增益0/16，不能认为计时隐状态问题已修复，也不能证明它是唯一原因。单纯EMA旧权重0.55/0.65/0.75对照正常28/28/12（32例），扰动7/5/1（16例），增强滤波不是可接受修复。

## 倒地起身及跨技能诊断

原起身ec665…，正常六倒姿48/48、受扰48/48；开启额外自碰撞/目标clip诊断也是各48/48，其最后3秒两膝多余行程正常2.790/2.224°、扰动2.848/2.262°，max-frame p95正常0.062/0.050°、扰动0.096/0.075°。原坐姿和当前v3坐姿(膝±1.25、髋∓0.4579)额外sit/stand两seed各4试验分别16/16；v3初始化仅改诊断metadata副本，原权重没改。以上是小样本诊断，accepted=false，不是实机起身验证。

getup-sit双专家直接组合e330…：正常15/32（循环均失败），扰动/低高增益全0/16，不部署。27组冻结实测传感器、只递推getup动作历史都收敛，last20差值RMS最高4.55e-7rad，因此没有独立历史自振的证据；不能用冻结输入结果替代物理闭环验收。

## 扩大示范约束后的监督修复

原保留锚只有约3000帧/3条轨迹，现将四条件的成功完整示范纳入保留，失败轨迹不充当专家：62条，seed8111训练23250帧，seed9001整条留出35000帧。同源85d7 actor，保持ONNX图及normalizer/动作合同，每档300步，权重1/3/10。留出actor偏离约0.0029/0.0052/0.0088rad；拟合内实测相邻变化0.04696rad降到0.02973/0.02271/0.01749，不能代表新实机改善。四条件闭环回归已全部结束，各档都丢失成功trial，不能部署。

## 专项与门控修复最终结果

冻结全部hidden层与其余12输出行，只训练两膝最后线性层；每档500步，冻结参数逐值校验。角速度门控沿用同一hidden层/归一化，角速度小时输出严格等于原actor，修复增量连续渐入；2913输入核对误差最高4.77e-7rad，其余关节输出严格一致。不是丢弃IMU、修改服务EMA或实机速度保护。仍不能保住全部成功例，所以排除全部候选。

|候选|正常32|扰动16|低增益16|高增益16|
|---|---:|---:|---:|---:|
|扩大示范全actor w1/3/10|28/28/28|7/5/4|10/2/0|13/13/13|
|膝输出 w1/2/3/10/30|28/28/28/28/28|9/9/9/7/5|12/11/12/6/1|13/13/13/13/13|
|门控3-0.75～1.5rad/s|28|8|12|13|
|门控3-1～2rad/s|28|8|10|13|
|门控10-1～2rad/s|28|8|6|13|

膝输出w1正常/扰动成功总数未减少、扰动多余行程减少5.05%，但低增益丢了stand_cycle/9001/3并补了另一成功例，总数12仍有退化，不能宣称已通过。w2扰动多余行程少12.05%，低增益丢2例。直接getup-sit组合用当前位置重建EMA并清raw历史的诊断，正常仍只有15/32且全部循环失败，不能靠切换清历史修复。

## 下一轮已准备，尚未启动

Upright奖励对照保持WeakProcess的全部动作、观测、命令和物理，只加端点倾角cost：参考ramp完成后，超过12°的倾角按10°归一化平方、weight−2；过渡ramp中这个成本为0，避免阻止必要的前倾起身。阈值是训练实验参数，非固件/实机保护。3项测试已通过，/home/leiwu30/microduck-training/sit-upright-control-20261010准备完整，launch脚本启动前发现GPU6约4.8GiB/100%被别人使用而拒绝，没有launch.json或训练PID，不打断别人任务。准备的driver会先64环境5轮smoke/导出检查，再1024环境100轮四条件筛选。每15分钟当前线程接续检查已创建（automationId=automation），安静等待资源，合格后独立回归/验收才部署；失败继续分析优化。若新奖励仍退化，下一可证伪方向是旧critic与监督修复actor/新目标不匹配，尚未做因果实验，不能当作根因。

完整证据包含full-followup-evidence.tar.gz（服务器状态参考/全部成功示范/修复与筛选）、此前v2/ablation/getup归档，source/normalizer核验和失败结果保留。历史启动说明保留在README下方；最终证据覆盖当时“未完成”状态。
'''
(folder/'diagnostics.md').write_text(report,encoding='utf-8')
readme=folder/'README.md';content=readme.read_text(encoding='utf-8')
prefix='> 最新结果见 [diagnostics.md](diagnostics.md)：1000轮与分项对照已结束且失败，没有部署；监督/专项/门控修复也已完成且失败；下一组倾角奖励对照等待空闲GPU。以下是启动时的历史说明。\n\n'
if not content.startswith('> 最新结果见'):readme.write_text(prefix+content,encoding='utf-8')
print('recorded completed failures and bounded follow-up')
