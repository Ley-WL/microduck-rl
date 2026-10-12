"""Record user authorization, offline scope, and current finite job."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
folder=root/'artifacts/process-repair-v2-20261010'
description='''> **2026-10-10 用户授权持续优化/条件部署（最新覆盖）：** 用户离开期间要求效果好直接部署替换，效果不好继续分析优化，并检查倒地起身是否同样振荡；覆盖历史“不自动续训/部署”，不表示无人托稳时自动发实机动作。上一轮失败后新建SitStandProcess-v2：实际EMA两膝目标变化/加速度以0.05rad诊断尺度归一化，过程惩罚−0.15；真实相邻输入平滑权重0.3→3，on-policy保留2→0.2，仿真示范保留10。切换扰动采样0.25→1、reset后首次interval同步新指令；BAM增益每world单次采样替代14关节均值，恢复实际±20% P缩放/±10%反电动势速度缩放范围，仍非HD1910固件D辨识。新任务保留旧任务及v3/61维/EMA0.45/HOME/限位/原始历史；新4项配置/过程/增益检查、先前相关20项及64环境5轮/160步1022次扰动短测通过，归一化逐值不变。GPU6/4096环境另1000轮launcher1899602已启动，未完成；名义/扰动/高低增益回归及过程门后再筛选，不部署失败候选。板上哈希只读核对仍sit-smooth-v1坐站85d7…、起身ec665…。倒地起身CPU两seed×6倒姿×4试验正常48/48、物理扰动+一帧速度延迟48/48，最后3秒两膝单帧p95正常0.069/0.056°、扰动0.085/0.067°，本批未见坐站级振荡，暂不改其权重；非实机运动/正式200例验收。证据平级RL artifacts/process-repair-v2-20261010/。

'''
for p in [root/'AGENTS.md',root/'docs/server-training.md',root/'docs/hd1910-stand-training.md',
          root.parent/'microduck-observer/AGENTS.md',root.parent/'microduck-observer/docs/实测记录/README.md',
          root.parent/'microduck-observer/docs/实测记录/03-舵机实测.md']:
    text=p.read_text(encoding='utf-8')
    if not text.startswith(description):p.write_text(description+text,encoding='utf-8')
readme='''# 2026-10-10 过程优化v2与倒地起身检查

用户授权效果好直接部署替换，失败继续分析优化；不在无人照看时启动实机运动。新任务Mjlab-SitStandProcess-Flat-MicroDuck-HD1910独立目录/home/leiwu30/microduck-training/sit-process-v2-20261010，旧任务/既有模型保留。GPU6/4096环境，最多1000轮，launcher1899602。当前仍训练，不能称已改善/部署。

证实的训练覆盖问题：旧实测相邻输入平滑项在固定观测上优化，未保证闭环；原过程加速度成本日志约−0.0097、腿姿+3.6006，不能认为过程得到同等约束。原扰动每interval回调概率0.25，与CPU挑战连续回调施加不同。旧randomize_delayed_actuator_gains将14份均匀样本取平均，实际P缩放标准差理论约0.0309而非均匀±20%的0.1155，且难覆盖边缘。BAM kd_scale调整电气反电动势项，不是实物D20反馈。

新候选以实际EMA+HOME+clip后的两膝目标变化和加速度cost，0.05rad仅归一化尺度（不是固件限速），权重−0.15；真实相邻输入平滑项3、示范保留10、on-policy保留0.2。保持LR1e-5、2epochs、4minibatches、固定actor归一化，原始上一动作观测。扰动概率1，reset回调标记unsynced、首个interval同步新指令，避免reset被误认命令切换；首次施加仍在interval检测到命令切换后，CPU边沿即刻施加，两者有interval相位差。执行器P/电气速度缩放按world真正单样本均匀抽取，当前范围不冒称实机参数映射或完整动力学对齐。

新4项配置/目标反转/reset/采样范围测试通过；相关组先前20项通过（其中当时新过程测试3项）。64环境5轮正规训练/导出exit0，额外64环境160步1022次扰动/64worlds，finite=true、归一化逐值保持。第一次smoke-before-gain-fix为改采样前历史，最终smoke为改后版本，长训练启动前校验所有训练源哈希和smoke相同。

CPU筛选使用同服务器同初态逐trial对照，名义32例、过渡扰动+一帧速度延迟16例、P缩放0.8/1.2各16循环例。保留每个原成功例、检查两膝多余行程与单帧跳变。检查点100/500/999导出，最后再跑GPU robust32worlds×2seeds×4scenarios；通过相对筛选仍不能自动写成正式200例/每组验收或实物验收，必须另做扩大样本部署审查。

## 当前倒地起身检查

板上及本地getup SHAec665f3121d8d45da0abd3d5792955d0aa46ba39f2897e3bbe01788fda9770f4，只读核对匹配。CPU六倒姿、两seed8111/9001、每组4例，正常48/48、物理扰动+一帧速度延迟48/48，初态逐例相同。扰动在起身1.0～1.8秒对真实物理俯仰角速度施加增量，未伪造gyro观测。分0～3/3～6/6～9/9～12/12～15秒记录全关节目标行程，区分起身多阶段动作和站稳振荡。

最后3秒两膝多余行程均值：正常2.80/2.19°，扰动3.10/2.44°；单帧最大变化p95：正常0.069/0.056°、扰动0.085/0.067°。本批未发现坐站那种明显振荡，暂不改起身权重；不代表实机起身已验收或所有受扰条件无问题。getup-process/comparison.json及getup-process.png为证据，图已核对；无需削弱正常起身的反馈控制。
'''
if not (folder/'README.md').exists():(folder/'README.md').write_text(readme,encoding='utf-8')
print('recorded process-v2 start and 96 getup trials')
