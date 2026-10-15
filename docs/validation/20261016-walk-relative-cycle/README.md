# 相对速度奖励独立筛选与价值校准

v64在新254151/254157种子、0/.03/.04/.06/.085/.2速度、yaw0±.45下冻结v55与v61，两版各144/144，job0/source-check通过。完整归档SHA `2d310a5f6378f6fad89360937575460410bb8c952d17d3de79f0f6b207c5b3cd`；reward-time bias重算最大1.53e-6，原容差通过，无失败world剔除。

预先记录的方向分离诊断：速度误差改善>.01m/s且yaw不恶化>.05rad/s；yaw误差改善>.05rad/s且速度不恶化>.01m/s。速度旧35/39，cycle35/39，relative36/39；yaw三者均21/21，两seed均未掉正确排名。但原v63b数据中一seed的yaw旧14/15→relative10/15，不能把holdout一条改善当显著或全面收益。保留每条rankingPairs以便查看代价取舍。

六个阶段窗口仅涵盖部分整个运动循环，不能把其平均分相加冒充完整折扣回报。v64仅支持继续小范围价值校准诊断，不证明步态改善，未批准部署或自动actor训练。

## v65：实验奖励与完整回报价值校准

新walk_relative_cycle_context继承continuous命令分布和86维critic上下文，actor仍61维，PD/物理/动作及其余奖励保持。两tracking项与bias使用tau=clamp(.1/|vxy|,.7,4.)，zero/pure-yaw .7。bias的纵向误差尺度clamp(|cmd_vx|,.025,.1)，zero/pure-yaw .1；横向.1、yaw.4和weight−4、cap25保持。与离线relative重放相同，不宣称旧奖励不变。

实际实现新增配置/符号/上限/partial-reset测试，并用v64记录的72world×1500帧reward-time输入验证torch计算与独立float64重放一致。新奖励不能继续使用旧critic的合格结论。

独立v65从v55step500 actor与v60b500 critic开始，actor/std.005/normalizer冻结，旧Adam不恢复；critic只是待重新校准的起点。255101/255107×96world×60秒完整回报，前30秒监督、每组第4world留出，5/100/500更新；之后新255151/255157独立价值门。仅critic训练，结果待核验；未启动PPO动作优化或部署。

完成后下载walk-relative-cycle-critic-v65-20261016，按v65-critic-spec.json运行check_walk_mc_dataset.py和check_walk_critic_readiness.py；值域/归一化/actor精确导出、source-check及完整归档核验后再决定是否有限动作训练。即便价值门通过，必须重新CPU/GPU宽速度/转向/停止/步幅和抗扰验收，尤其观察已知低速yaw取舍。

最新：launcher313029，context-test全部通过，真实轨迹相对cost重放最大误差1.48e-6；进入collect。证据v65-test-output.txt。
