> **2026-10-16 v59完成，未合格：** job0/source-check/完整归档752e78c845c7b5e4f8181714a25d2ed02c5a470bf2ca0fbd7a0dd567bde41e67核验。监督100/500更新，RMS .007742→.006441、停止RMS .002692→.002257，不能替代闭环验收。CPU原/100/500各12/12，逐轨迹初始qpos/首velocity/全command精确配对。500步+.05实际.05344/.05840、脚步5.465/6.065cm，但yaw−.1172/−.1050；+.075实际.09383/.09278仍偏快。GPU原191/192、候选190/192，250157丢原成功world17（−.05/yaw0），两个seed内共同成功world的指令hash均一致，失败world保留不筛掉。GPU+.035实际.0260→.0168、+.05 .0542→.0433，低速启动退化；+.075 .0672→.0835且足行程3.961→5.450cm，仍有速度取舍。零指令漂移/若干停止改善，不能抵消新摔倒。停止增加监督轮数，不部署。当前无活跃训练；下一应先诊断连续低速采样及新actor价值校准，不能直接继承旧critic当有效，更不能将后退偏航脉冲作为好示范。PD/硬件/61→14/无额外运行时反馈保持，目标未完成。详见docs/validation/20261016-walk-joint-pulse/v59-results.json。

# 同环境脉冲示范与CPU/GPU联合模仿

> **2026-10-16 v57b/v58教师物理可行性完成/v59联合模仿活跃：** v57b原/脉冲各32/32，前进.035/.05/.075/.1实际.0333/.0526/.0797/.1013，停止多数改善；反向偏慢并yaw+.060/.083/.117/.151，反向脉冲不作示范。job0/source-check/归档7751d298992009734ba2a46ca335761b6aca6aaf1970eb9c893bb60e6edca28f。v58新249101/249107原/教师各32/32，新增按前进/反向分别记足部行程并核对分相总和；前进步幅5.316/4.505/5.471/5.840cm，不能把两方向总均值冒充前进步幅。保存candidate前800帧（正向+首停）和baseline1500帧；原始61维obs、raw目标、live、整world验证split，新增脚本核对全部形状/有限值/失败不重入/phase统计和指令EMA。校验先发现tuple/JSON list表示及actor指令为既有smoothed_twist .2而非原始命令，已按源码复算逐帧通过，无数据重采/无阈值放宽；job0/source-check/归档03cd23795f036bfc209f5ec641ab830ec2ab35055820764eebbc4687e466b1f8。v59独立walk-joint-pulse-v59-20261016 launcher4125370活跃：v55step500/空监督Adam，56600帧CPU+GPU示范、41084帧v45保留obs（GPU验证world排除、正向低速从anchor排除）、596CPU停止恢复帧；保留61→14/无clock，5/100/500预算，新250171/177 CPU六速度及250151/157 GPU含yaw筛选，结果待核验。无PD/实机/部署，目标未完成，勿重复启动。

v56聚合低速指标混合了正反向和转向，不能单独证明正向大步完全丢失。例如+.05/yaw0的正向平均速度从.0618到.0557，反向从-.0400到-.0318。v58新增分方向足部统计，解决混合指标的诊断局限。

v59 GPU candidate只收集前800步，全部原始actor输入含原来的指令EMA，私有pulse时钟、critic观测及实际根速度都不加入actor输入。训练数据中GPU phase字段=-1，仅是未记录标记，训练器只读取observation/target/validation/world。每case第4个world验证，其余训练，原CPU划分保持；GPU baseline仅训练world进入retention且排除正向低速运动区间。GPU/CPU停止状态、后退与转向仍需闭环验收，retention不是成功保证。

独立校验最初把actor输入命令误当成原始命令，已依据walk_surface_transfer.smoothed_twist的old+.2*(current-old)重新计算，CPU/GPU浮点差采用atol5e-8/rtol1e-6；不是将明显不同输入强行判为相等。

## v59结果解释与下一步

CPU与GPU的低速表现不同，不能根据单环境选优：CPU .05已较接近目标，GPU .05反而偏慢；.075两边均偏快。相同速度的直行/后退/转向须分别报告。GPU失败world仍计入聚合，其失败后指标可能为零，不能把聚合指令变小解释为真实目标改变。

这轮说明继续降低模仿误差并不保证更好的闭环速度。下一步先检查训练的速度覆盖：现有离散SPEEDS不含.035/.075，低速偏重±.05。拟探索覆盖连续低速且保留零速、倒退、原地转向的采样，再对选定actor重采完整回报并校准critic；价值验证通过后才小预算PPO。不得把此计划描述为已启动或已有改善。旧v44价值网络来自v28分布，不能直接认定适用于v55/v59。

复算命令：`python scripts/audit_walk_joint_pulse.py artifacts/walk-joint-pulse-v59-20261016/downloaded/evidence --output docs/validation/20261016-walk-joint-pulse/v59-results.json`。所有候选仍不合格，未部署，未改PD。
