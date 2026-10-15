# kissqy 行走标称仿真对齐（2026-10-15）

显式比较配置 `--physics-profile kissqy`，不改变 HD1910/当前 STL 训练默认值或实机 PD。来源为本地 kissqy/Microduck 提交 `7a52d297a9adfe6d2315481953dc64060252d142` 的 walk/deployment-contract.json；完整快照位于 scripts/contracts/kissqy-walk-7a52d29.json。未声称是作者当前实机设置。

## 对齐内容

- 总质量 0.82 kg；15 个刚体质量、质心、惯量、装配变换。
- 14 个驱动关节及 14 个被动齿隙关节；齿隙 ±1°，被动阻尼 0.01、惯量 0.001；电机附加惯量 0.0017819049974475338。
- 15 个舵机壳体碰撞体及 2 个脚底；沿用本地 STL，并逐个验证编译后位置、姿态、尺寸与合同一致。壳体 condim1、脚底 condim3；摩擦与接触软约束采用合同值，关闭其余机器人碰撞体。
- 物理步长 5 ms、每 4 步推理一次（50 Hz）；implicitfast、Newton、30 次迭代、50 次线搜索及合同容差。
- HD1910 m6 参数文件保持 SHA 校验；BAM kp5、7.4 V、零电压下降为标称值，不表示已经辨识实机 P5/D0。
- 延迟固定 4 个物理步（20 ms），位于供体 3–6 步训练区间；此配置不另叠加旧 8 ms 传输延迟。齿隙编码器读 motor+passive，摩擦与反电动势读 motor 速度。

仍使用 Observer 当前执行逻辑（动作 scale1、头/腿新权重 .5/.7、指令 EMA .2、上一反馈周期速度）。初始站姿、理想 IMU 和编码量化是本诊断假设。没有复现 GPU 训练随机地形、摩擦/质量/电压随机化、随机推扰与随机延迟；完整范围保存在合同快照中。本次是标称物理对齐，不是训练环境逐项等价。

## 验证与结果

运行 `scripts/test_kissqy_walk_physics.py`：3 项通过，包括主动改变脚底几何/齿隙后必须拒绝的回归测试。旧 local+backlash 路径的未定义变量也已修复，并通过静止冒烟。

18 组：前后指令 ±0.2 m/s × yaw 0/±0.6 rad/s × 3 初态，12 秒；2 秒起步、4 秒叠加转向、9 秒停止。全部未触发 60° 跌倒门，最大倾角 5.236°。直行 4–9 秒世界 x 均速约前进 .0974、后退 −.0560 m/s；5–9 秒实际世界 yaw：零转向 −.181 至 −.071，正 .6 指令对应 +.225 至 +.307，负 .6 对应 −.603 至 −.546 rad/s。仍有偏转和左右不对称，不能把不倒当作跟踪良好。

证据：artifacts/kissqy-physics-20261015/{comparison,tracking}.json 及各 yaw 子目录的 manifest、summary 和逐帧状态。本次未连接硬件、未部署、未训练或修改权重。

复现（仓库根目录）：

```powershell
.venv-preview/Scripts/python.exe scripts/evaluate_walk_runtime_cpu.py --onnx ../microduck-observer/models/xgoduck_walk.onnx --physics-profile kissqy --output artifacts/kissqy-new-run --speeds .2 -.2 --seeds 130031 240017 310019 --trials 1 --seconds 12 --turn-rate .6
```

输出目录须不存在；省略 physics-profile 时保持原 local 配置。
