# MicroDuck HD1910 训练项目

本项目使用当前本地 `robot/replica` STL/MJCF 和 HD1910 BAM M6 动力学。所有已注册的 MicroDuck 任务都使用这套硬件；旧任务 ID 仅保留名称兼容，不再选择官方 XL330 参数。

## 视频展示

<table>
<tr>
  <td width="33%" align="center"><video src="https://github.com/user-attachments/assets/c5ef2333-d38d-456e-afb6-89fb5f8b2f60" controls></video></td>
  <td width="33%" align="center"><video src="https://github.com/user-attachments/assets/eb1ef356-b40e-4d24-ab92-f37a1b45dcf1" controls></video></td>
  <td width="33%" align="center"><video src="https://github.com/user-attachments/assets/1ee32755-6e41-4c7c-9f7c-88e3762b8e5a" controls></video></td>
</tr>
</table>

## 已训练模型 · 2026-10-21

每次提交训练、微调或评估更新，默认同步本表。按训练版本汇总，中间检查点合并；失败版本保留结论。成功率均为对应场景仿真，实机反馈单独注明。

| 模型 / 版本 | 方法 / 轮数 | 结果与当前状态 | 模型 / 配方 / 测试 |
|---|---|---|---|
| **行走 Observer 5000** | 官方配方 + HD1910 + Observer限位，5000轮 | **当前采用**；30输入480/480未倒，用户实机反馈好；小幅原地左转仍弱 | [模型与结论](models/walk_observer_hd1910_5000/README.md) · [配方](docs/walk-observer-official-training-20261020.md) |
| 行走 Observer 10000 | 在5000轮基础上续训至10000轮 | 已完成，未全面超过5000轮，未替换 | [当前选型](docs/walk-observer-official-training-20261020.md)；本地 `artifacts/walk-observer-10000-20261020/` |
| 行走原配方 HD1910 5000 | 保留原训练方法、更换硬件参数 | 曾部署，已被Observer限位5000替代 | 本地 `artifacts/walk-original-hd1910-20261020/` |
| 行走转弯 / 偏航修复 / 锚定 | turn、yaw-repair、anchored多轮实验 | 历史候选；旧训练入口已清理 | 本地 `artifacts/walk-{turn-training,yaw-repair,anchored}-*/` · [整理说明](docs/walk-observer-official-training-20261020.md#代码入口整理) |
| 行走慢走 / 摆腿 / 编码 / 延迟 | 早期行走与Encoder、Delay16适配 | 历史候选，存在回归失败；已由现役版本替代 | [历史记录](docs/server-training.md) |
| **金鸡独立右脚 3000** | 公开CycleHard从头训练，3000轮 | **仅右脚部署试用**；右48/48、左0/48，实物效果待测试 | [模型与结论](models/flamingo_right_hd1910_3000/README.md) · [配方](docs/flamingo-author-scratch-20261021.md) |
| 金鸡独立短微调 | 原ONNX迁移，100轮 | 保持61/96，低于原版90/96，不采用 | [方法与结论](docs/flamingo-cycle-hd1910-20261021.md) |
| 金鸡独立课程迁移 | ONNX迁移 + 公开课程，3000轮 | 保持0/96，不采用 | [方法与结论](docs/flamingo-published-curriculum-20261021.md) |
| 早期金鸡独立 | HD1910 Flamingo | 历史模型，已替换 | [模型合同](https://github.com/Ley-WL/microduck-observer/blob/feat/vision-camera-npu/models/flamingo.metadata.json) |
| 后抬脚 v27 / native / gated / v62b | 后抬脚与运行合同适配；native/gated共享权重 | 历史试用版本，未获得稳定实物结论，已替换 | [模型与合同](https://github.com/Ley-WL/microduck-observer/tree/feat/vision-camera-npu/models) |
| **十拍舞蹈 v5** | 大幅动作 + 抗扰续训1000轮 | **已部署试用**；正常64/64、训练扰动58/64、压力61/64，抗扰未全通过 | [配方与结论](docs/dance10-training.md) · [模型](https://github.com/Ley-WL/microduck-observer/blob/feat/vision-camera-npu/models/dance10.onnx) |
| 十拍舞蹈 v4 | Expressive大幅动作，800轮 | 正常64/64、扰动55/64、压力57/64；由v5替代 | [训练记录](docs/dance10-training.md) |
| 十拍舞蹈 v1 / v2 / v3 | 初版、稳定性微调、恢复训练 | v1正常/训练扰动128/128；v2压力有退化；v3旧压力63/64、新种子仍失败 | [训练记录](docs/dance10-training.md) |
| **机械限位倒地起身** | GetupMechanical，HD1910/当前限位 | **已部署**，用户曾反馈能起身；仍有实机反复失稳记录 | [历史与诊断](AGENTS.md) · [模型](https://github.com/Ley-WL/microduck-observer/blob/feat/vision-camera-npu/models/xgoduck_getup.onnx) |
| 倒地起身 v1–v3 / 4750 | 分阶段训练与头颈修正 | 4750轮完成仿真验收；后来换成机械限位版本 | [训练与验收](docs/server-training.md) |
| 坐站 5000 | 双向坐站训练 | 曾通过仿真，实物顶压/磕头失败，已替换 | [训练与实物差异](docs/server-training.md) |
| 坐站接触 / objective-v2 | 接触与目标奖励修正，各1000轮 | 双向循环失败，不采用 | [历史结果](AGENTS.md) |
| 坐站双专家 / 单actor种子 / PPO | 专家组合、监督初始化、1000轮微调 | 种子CPU循环31/32；PPO最终0/32，保留失败结论 | [历史结果](AGENTS.md) |
| 坐站平滑 / Transfer / Process | 平滑监督与过程约束微调 | 试用/筛选存在抖动与旧成功退化，未完成稳定验收 | [历史结果](AGENTS.md) |
| 坐站 Upright / Capped / LowExplore / CriticWarm / ArrivalGate | 端点、探索、价值预热、门控对照 | 各轮筛选仍丢旧成功，未采用 | [历史结果](AGENTS.md) |
| 坐站 HeadClearance / Pitch / FixedLR | 头部接地、前倾代价、固定学习率，各20轮对照 | 未消除扰动磕头或存在动作回归，不采用 | [历史结果](AGENTS.md) |
| 翻滚机械合同 Control / Repair | 固定学习率，各20轮 | 候选254/256、原255/256；无有效提速，不采用 | [配方与结论](docs/roulade-training-20261012.md) |
| 站立/恢复 v1–v3 | StandHold、恢复与连续推扰 | 历史探索，静止时头姿及目标振荡不合格 | [训练记录](docs/hd1910-stand-training.md) |
| 静稳 v4 / v4b | 撤力静止；v4终止，v4b完成2000轮 | v4b静止无倒地但头部偏差约25.5°，不采用 | [训练记录](docs/hd1910-stand-training.md) |
| 静稳头姿 v5 / v6 | v5修正2000轮，v6再续3000轮 | v5曾部署试用；两版抗扰均未全通过，v6未采用 | [训练与部署](docs/hd1910-stand-training.md) |

原版行走、官方Flamingo、拾取、踢球和原版翻滚是外部基线/导入模型，不记作本项目新训练成果。旧记录中的“运行中”以同实验较新的最终记录为准；没有最终证据的版本不补写成功率。本地 `artifacts/` 与服务器完整checkpoint未入库，ONNX不能替代PPO续训checkpoint。

## 唯一硬件入口
本仓库是基于 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) 的 HD1910 适配与训练实验分支。

- `src/mjlab_microduck/robot/hardware_profile.py`：HD1910 参数路径、P5、训练电压 7.4–8.0V、压降增益 0–0.2、有效电压下限 4.0V。
- `src/mjlab_microduck/robot/hd1910/1910_m6.json`：LuwuDynamics 公开的 1910 M6 参数，来源与许可证见同目录 README；不是本机重新辨识结果。
- `src/mjlab_microduck/robot/replica/`：当前装配 MJCF、STL、关节和碰撞变体。
- `src/mjlab_microduck/robot/hardware-inputs.json`：上述几何/执行器输入的 SHA256 清单。

`xl330.stl` 是沿用的 CAD 外壳文件名；它不参与舵机参数选择。现有质量和惯量来自本地 MJCF，不能用 STL 的外形声称已完成 HD1910 整机质量辨识。训练不再从供体文件覆盖质量、惯量或装配变换。

## 常用入口

在本目录使用 Linux/CUDA 训练环境；本次整理不自动启动训练。

```bash
uv sync
uv run python scripts/audit_hardware_profile.py --configs
uv run list-envs

# 行走入口先64环境5轮冒烟，通过后自动进行4096环境5000轮训练
uv run python scripts/server/train_walk_observer.py launch
uv run train Mjlab-GetupMechanical-Flat-MicroDuck-HD1910 --env.scene.num-envs 64 --agent.max-iterations 5

# 导出保留内嵌观测归一化；明确指定本地 checkpoint
# 行走由上述训练入口自动导出；模型见 models/walk_observer_hd1910_5000/
```

常用任务：`StandHold`（站立保持）、`train_walk_observer.py`（当前行走配方）、`GetupMechanical`（现行机械限位起身）、`SitStandContact`（坐站接触），完整 ID 以 `list-envs` 为准。任务各自的奖励、PPO、动作 EMA、限位和观测合同继续明确区分，不能把一种技能的动作滤波随意套到其他技能。通用 PPO/奖励算法保留，移除的是旧硬件默认参数和默认官方教师权重。

新训练默认使用本地 TensorBoard；实验名有 `hd1910` 标识。默认不下载官方教师，不自动推送权重。迁移学习必须显式选择兼容 checkpoint，旧权重不会因为更换配置自动成为新硬件训练结果。

## 验证和预览

```bash
uv run --with pytest pytest tests/test_hardware_profile.py tests/test_hd1910_stand_cfg.py tests/test_infer_policy_bam.py
uv run python scripts/infer_policy.py --walking /path/to/policy.onnx
```

CPU 推理入口也使用同一 HD1910 JSON；`--no-bam` 已拒绝，避免退回旧 XML PD。Windows 可使用现有 `scripts/preview_hd1910.py --policy ...` 预览带配套 metadata 的本地策略。预览必须匹配对应策略的动作合同，不能代替实机验收。

历史实验、原始权重和证据保留在 `artifacts/` 和历史记录中。上游说明见 [参考归档](docs/reference/upstream-readme.md)；旧 `robot/microduck`、XL330 台架和 XML-PD `sim/body_server.py` 仅作历史参考，不是当前训练入口，`duck-body` 命令不再发布。具体清理范围与验证见 [整理记录](docs/hardware-profile.md)。

代码沿用 Apache-2.0；模型/STL 的原始来源与许可声明继续保留，详见归档 README 和 HD1910 参数来源文档。
