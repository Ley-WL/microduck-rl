# 2026-10-14：训练项目统一 HD1910 与本地 STL

用户要求：去掉官方训练参数，完全使用 HD1910 和现在的 STL 模型。本次仅修改本地训练项目；未启动训练、访问主板、部署或推送远端。

## 当前入口与来源

`robot/hardware_profile.py` 统一提供 HD1910 JSON、P5、7.4–8.0V 训练电压、0–0.2 压降增益、4.0V 下限与默认 3–6 仿真步命令延迟。技能明确指定的域随机化、14/16ms 延迟对照、动作 EMA/scale、机械限位、61→14 接口继续属于各自任务合同，不能混成一个配置。3–6 是仿真步数，不是实机已辨识延迟。

所有 69 个注册 MicroDuck 任务的训练/预览两种配置（138 组）现在使用 `robot/replica` 模型与 HD1910 执行器；旧任务 ID 为兼容名称。注册检查拒绝 XL330、其它 JSON、P200 或旧电压下限。普通和回差执行器均支持 Feetech 目标状态初始化/部分环境 reset；回差版本保留输出侧编码器反馈。

输入清单 `robot/hardware-inputs.json` 保存 64 个 XML/STL/执行器文件的 SHA256。训练 runner 与正规导出检查清单；新训练日志写 `hardware-profile.json`。ONNX 附加的是“本次导出配置”来源，不把老 checkpoint 冒称为新硬件训练。

## 删除的生效路径

- 共用执行器不再加载官方 XL330 M6、P200、6.5–8.2V 和 6V 下限。
- 当前 replica 的 6 个机器人 MJCF 及 `joints_properties.xml` 删除旧 XL330 PD 模板，剩下零增益结构占位；BAM 在组装时提供扭矩、电压、摩擦和转子惯量。CAD 再导出也运行 `strip_legacy_actuators.py`。
- 行走与翻滚迁移工厂不再从供体合同覆盖质量、惯量、质心或装配变换；CPU 行走诊断同样使用本地 MJCF。旧 `--body-contract` 仅接受为兼容参数，报告 `bodyContractApplied=false`。
- VelStand 默认关闭官方教师；蒸馏默认不含官方 W&B 权重地址。显式迁移仍须自行指定兼容 checkpoint。
- CPU `infer_policy.py` 使用统一 HD1910 JSON，支持 Feetech reset；拒绝 `--no-bam`；去掉源于 XL330 P200/0.55 的录制保持增益换算。
- 新 manifest 标注 `servos=hd1910`。默认本地 TensorBoard 与 hd1910 实验名。移除发布的 `duck-body` 命令，旧 XML-PD body server 仅作参考。

通用 PPO、奖励、归一化与观测实现保留。它们是任务算法，不是 XL330 的电机参数。历史脚本/日志/权重与上游许可来源保留，不删除试验失败证据；旧的官方模型目录、台架代码和 [上游 README](reference/upstream-readme.md) 仅供追溯，不在当前注册任务链路中。

## 几何保持与验证

所有 STL 字节未改。清理 XML 前后逐数组检查：body mass/inertia/ipos/iquat/pos/quat、关节 axis/pos/range、geom pos/quat 和 mesh vertices 完全一致。本地普通/回差模型原本的细小质量差仍保留，未伪装成统一实测质量。

保护性跌倒任务需要舵机外壳接触传感器，但本地模型该外壳原本只有视觉网格。只在该工厂添加同网格的地面接触副本（零附加质量），不改 STL 或刚体惯量、不启用外壳重叠自接触。

- 185 项软件检查通过，2 项跳过：默认官方 warm-start 已关闭；实际 GPU rollout 因本地 CPU-only Torch 跳过。
- 69 个任务 × train/play = 138 配置全部解析/编译，确认本地网格路径、14 个执行器、HD1910/P5/4V、50Hz。
- CPU BAM 实际步进、目标 reset、参数来源、发布 manifest 与相关任务回归通过。
- Python 编译和 `git diff --check` 通过。
- 未运行 GPU 64×5 训练/ONNX 冒烟、未实机验收；后续新训练先执行该阶段。

证据：`artifacts/hardware-cleanup-20261014/{summary.json,audit.json,geometry-preservation.json,tests.log}`。第一次回归发现保护性跌倒外壳传感器缺失，已修复后复跑；另一个旧 warm-start 测试替身缺少 runner.alg，已补齐真实对象接口。

`xl330.stl` 仍是当前 CAD 的外壳标识，不能据此宣称已测量 HD1910 的外形和质量。HD1910 M6 是 LuwuDynamics 公布参数，当前质量/惯量是已有本地模型值，均不等于本机重新辨识完成。
