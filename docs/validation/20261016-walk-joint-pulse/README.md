# 同环境脉冲示范与CPU/GPU联合模仿

> **2026-10-16 v57b/v58教师物理可行性完成/v59联合模仿活跃：** v57b原/脉冲各32/32，前进.035/.05/.075/.1实际.0333/.0526/.0797/.1013，停止多数改善；反向偏慢并yaw+.060/.083/.117/.151，反向脉冲不作示范。job0/source-check/归档7751d298992009734ba2a46ca335761b6aca6aaf1970eb9c893bb60e6edca28f。v58新249101/249107原/教师各32/32，新增按前进/反向分别记足部行程并核对分相总和；前进步幅5.316/4.505/5.471/5.840cm，不能把两方向总均值冒充前进步幅。保存candidate前800帧（正向+首停）和baseline1500帧；原始61维obs、raw目标、live、整world验证split，新增脚本核对全部形状/有限值/失败不重入/phase统计和指令EMA。校验先发现tuple/JSON list表示及actor指令为既有smoothed_twist .2而非原始命令，已按源码复算逐帧通过，无数据重采/无阈值放宽；job0/source-check/归档03cd23795f036bfc209f5ec641ab830ec2ab35055820764eebbc4687e466b1f8。v59独立walk-joint-pulse-v59-20261016 launcher4125370活跃：v55step500/空监督Adam，56600帧CPU+GPU示范、41084帧v45保留obs（GPU验证world排除、正向低速从anchor排除）、596CPU停止恢复帧；保留61→14/无clock，5/100/500预算，新250171/177 CPU六速度及250151/157 GPU含yaw筛选，结果待核验。无PD/实机/部署，目标未完成，勿重复启动。

v56聚合低速指标混合了正反向和转向，不能单独证明正向大步完全丢失。例如+.05/yaw0的正向平均速度从.0618到.0557，反向从-.0400到-.0318。v58新增分方向足部统计，解决混合指标的诊断局限。

v59 GPU candidate只收集前800步，全部原始actor输入含原来的指令EMA，私有pulse时钟、critic观测及实际根速度都不加入actor输入。训练数据中GPU phase字段=-1，仅是未记录标记，训练器只读取observation/target/validation/world。每case第4个world验证，其余训练，原CPU划分保持；GPU baseline仅训练world进入retention且排除正向低速运动区间。GPU/CPU停止状态、后退与转向仍需闭环验收，retention不是成功保证。

独立校验最初把actor输入命令误当成原始命令，已依据walk_surface_transfer.smoothed_twist的old+.2*(current-old)重新计算，CPU/GPU浮点差采用atol5e-8/rtol1e-6；不是将明显不同输入强行判为相等。
