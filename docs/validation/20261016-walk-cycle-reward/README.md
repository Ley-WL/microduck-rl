# 低速步态周期奖励诊断

v62冻结v55/v61，各120/120存活，240个steady阶段，无失败world被剔除。job0/source-check及完整归档baff7a50862843ecd135a15356892c2a9ec89162a1d6dcde4780cb1bafb9e16a核验。

+.025直行实际原.03317/候选.00637m/s，总分10.0918/10.3447，候选5/8更高；tracking_bias候选少罚.2334、linear跟踪反而多奖.1106，air_time少.2313。+.035原.03477/候选.01470，总分10.0475/10.0291，分差很小。不能只增加训练轮，低速评分存在对少动的偏好；不是已证明唯一因果。

v63只改变两项tracking reward及tracking_bias的平均时间常数，tau=clamp(.1/|vxy|,.7,4.)，zero/pure-yaw仍.7；依据10cm进展周期而非外部运行时clock，PD/actor输入/物理/其他奖励不改。两个tracking误差与bias误差的状态一致性通过测试，未来critic已有误差字段可用，但当前没有训练。旧评分参考独立更新，仅为同轨迹诊断。停止历史会受运动阶段影响，不能宣称过渡奖励精确不变。

基础测试通过：低速周期平均正确运动优于停住、legacy逐值精确、快速/初始零指令/纯yaw保持、partial reset正确、停后误差消退、critic误差记忆一致。v63 frozen各120/120，job0/source-check、归档5fa87de74c1f004b566040baaf4a369f27c7ca7403ad22092127c60750f41b2b；同轨迹+.025旧原/候选10.1137/10.3264→新10.6802/10.6301，偏好候选5/8→2/8，+.035平均分差也转向原版，但逐world并非全改善，不能当训练验收。

独立bias重算未通过：把post-step记录速度用于reward会产生最大.123差异。已检查已安装mjlab ManagerBasedRlEnv.step源码：reward计算在forward刷新之前，root velocity等派生量滞后一个5ms物理子步；记录的motion在forward和CommandManager.compute之后。单纯移一帧command不能修复sensor-view差异，未放宽2e-5/1e-5容差，也未改物理。

v63b独立采集：CycleBias在reward调用内额外保存actual与command副本；只用于日志，不写入actor/critic或动作。冻结同模型/同seed/同奖励重跑，同时保存post-forward motion和reward-time输入。独立检查器以实际reward输入重算旧/新bias，校验command与post-step前一帧关系、未改变项逐值相同，保留失败world数。结果待核验，不得基于v63未核验结果开始训练。

完成后：

```powershell
.venv-audit/Scripts/python.exe -X utf8 .local/collect_walk_experiment.py walk-cycle-reward-v63b-20261016
.venv-audit/Scripts/python.exe -X utf8 scripts/check_walk_cycle_reward.py artifacts/walk-cycle-reward-v63b-20261016/downloaded/evidence --output artifacts/walk-cycle-reward-v63b-20261016/independent-bias-audit.json
```

再分别用analyze_walk_yaw_reward_audit.py默认和--legacy读取同轨迹评分，并比较所有速度/转向/正反/停止阶段；不能只挑+.025平均改善。无动作训练、实机或部署，目标未完成。

## v63b完成

job0/source-check通过，归档SHA `f53c594beae45afdfc3e3b8847cd378383bb6bd45b18c8b699d71842d68e7f62`。原/候选各120/120，所有240world的reward-time输入独立复算通过，最大误差9.86e-7，未放宽原容差2e-5/1e-5。command确实为post-step前一帧；速度需reward计算时值，不可拿post-forward派生量替代。其余奖励逐值不变，证据independent-bias-audit。

预先采用“速度绝对误差改善>0.01m/s，yaw误差不恶化超过0.05rad/s”作诊断筛选，不是完整模型验收门。v63共39条符合，旧/周期评分选对36/35；v63b相同seed再次运行44条符合，旧/周期38/39，说明少量排序改善并不稳定，不能靠+.025平均值宣布奖励已解决。

另在同v63b完整输入上只做离线评分，vx尺度由固定.1变为clamp(|cmd_vx|,.025,.1)，其他分量保持(.1,.4)，zero/pure-yaw尺度仍.1；周期均值保持。验证未更改尺度的帧逐值一致，原cycle bias按原容差重算通过。离线相对误差方案选对40/44，但属于这组数据上的假设筛选，不是独立验证，更不是新actor。代码replay_walk_relative_tracking.py，完整输出保留artifacts；未应用于训练环境。

当前无活跃任务。下一须新的holdout命令/seed检验评分排序及整个起停阶段取舍，保留大步奖励和零指令约束；证据支持才实现新奖励并重新校准critic，再有限PPO。禁止直接用旧v60b critic为修改后奖励背书。
