# 连续速度/转向覆盖与价值重新校准

v59联合模仿未合格，不再增加模仿轮数。原训练速度为离散档位，且至少50%重采样到±.05；这不直接证明偏差唯一成因，但未直接覆盖.035/.075，是可独立检验的缺口。

新任务walk_continuous_coverage只改指令采样，奖励、物理、PD、actor/critic输入合同保持。60%连续低速绝对值.025–.12m/s，20%连续较高速.12–.4，10%原地转向，10%精确零指令；速度正负对称，移动指令一半直行，其余yaw连续±.6rad/s，原地转向绝对值.15–.6。保留既有30秒前进/停/反向/停循环。现有步幅奖励仍给6cm距离信用，权重1，不能称此次已提高步幅奖励。

v60b冻结v55step500动作网络及归一化，std保持.005；价值网络以v45的86维网络为初始化，但明确尚未校准，旧Adam不恢复。连续样本的24组固定诊断命令见critic-spec.json，2seed×96world×60秒，监督前30秒完整折扣回报，每组第4条whole-world留出。5更新冒烟、100筛选、最多500更新，仅更新critic。随后两独立seed以新的24组低速/停止/转向命令检查原价值门，不放宽阈值。actor输出必须导出SHA精确等于v55。价值合格也不代表步态合格，不自动PPO或部署。

初始v60在测试夹具使用SimpleNamespace调用super时TypeError，job.exit=1、进程4191839消失，未仿真/拟合。改用实际命令子类测试partial reset后，独立v60b launcher8774已启动，采样分布/正负/直行/纯yaw/停止/部分reset及物理、奖励、观测完全一致测试通过；结果仍待核验。

终态下载后运行：

```powershell
.venv-audit/Scripts/python.exe -X utf8 .local/collect_walk_experiment.py walk-continuous-critic-v60b-20261016
.venv-audit/Scripts/python.exe -X utf8 scripts/check_walk_mc_dataset.py artifacts/walk-continuous-critic-v60b-20261016/downloaded/evidence --spec docs/validation/20261016-walk-continuous-coverage/critic-spec.json --output artifacts/walk-continuous-critic-v60b-20261016/dataset-audit.json
.venv-audit/Scripts/python.exe -X utf8 scripts/check_walk_critic_readiness.py artifacts/walk-continuous-critic-v60b-20261016/downloaded/evidence/critic-quality.json --spec docs/validation/20261016-walk-continuous-coverage/critic-spec.json --output artifacts/walk-continuous-critic-v60b-20261016/readiness.json
```

检查器保留旧默认路径，并已用v44完整历史回报与价值门复算通过。新数据将使用同阈值复算。完整回报检查改为一次解压奖励/终止数组，避免每帧重复解压。

后续仍须单独验证实际速度、转向、足部行程、停止、倒退、连续指令切换与抗扰。无实机操作、无部署，PD不变；目标未完成。

## v60b完成与v61动作微调

v60b job.exit=0、source-check通过，完整归档SHA `edb51bd950f935824e2f3ee453dcf895ee3404156f6f3c04867a653b6631aee0`。独立float64回报及whole-world分组复算通过。两独立seed各96/96，完整循环EV .605/.600，原10项价值门全部通过；留出RMSE从1.476降至1.343。actor/normalizer保持v55，不能称新步态通过。

v61独立启动，launcher72500。采用v55step500 actor与v60b mc-step500 critic，准备阶段逐tensor验证critic checkpoint的actor与v55相同；只加载actor/critic，空Adam、iteration0。保留旧奖励/物理/PD，连续命令分布生效，固定std=.005，不使用从.05起步的anneal。新增low模式验证200个rollout/update的精确log-prob一致性，并重跑旧fixed/anneal/resume测试通过。64×5冒烟通过才1024×200，anchor系数5、LR1e-5；之后新252151/252157、13速度×yaw0±.6×4环境，原v55逐例配对。当前冒烟在途，结果未验收，不能重复启动。

GPU通过也不是最终验收：必须CPU同初态宽速度、停止、连续换方向及抗扰，并看视频确认步幅。低速.025/.035/.05/.075和后退、原地转向分别统计，不能用全局均值掩盖退化。当前无部署。

最新：64×5冒烟/40Adam/finite/normalizer/正规导出通过，ONNX误差3.34e-6；已进入1024×200主训练。证据v61-smoke-audit.json，尚无闭环改善结论。
