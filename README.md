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

## Observer 当前使用的模型 · 2026-10-22

仅列当前服务使用的模型；历史训练、失败实验和备用权重不列入首页。

| 功能 | 当前版本 / 来源 | 状态与说明 | 模型 / 测试说明 |
|---|---|---|---|
| 行走 | HD1910 + Observer限位，5000轮 | 用户实机反馈好；保留当前版本 | [模型与测试](models/walk_observer_hd1910_5000/README.md) · [训练方法](docs/walk-observer-official-training-20261020.md) |
| 金鸡独立 | HD1910 CycleHard，3000轮 | 仅抬右脚；仿真48/48，实机出现软掉/重启，原因待查 | [模型与测试](models/flamingo_right_hd1910_3000/README.md) · [训练方法](docs/flamingo-author-scratch-20261021.md) |
| 十拍舞蹈 | HD1910 大幅抗扰 v5 | 已部署试用，抗扰仍有失败 | [模型](https://github.com/Ley-WL/microduck-observer/blob/feat/vision-camera-npu/models/dance10.onnx) · [训练与测试](docs/dance10-training.md) |
| 倒地起身 | HD1910 GetupMechanical | 已部署，仍有实机失稳记录 | [模型](https://github.com/Ley-WL/microduck-observer/blob/feat/vision-camera-npu/models/xgoduck_getup.onnx) · [合同](https://github.com/Ley-WL/microduck-observer/blob/feat/vision-camera-npu/models/xgoduck_getup.metadata.json) |
| 坐下 / 站起 | 基于 HD1910 重新训练，3000轮 | 用户实测通过；连续稳定0.5秒切回行走 | [模型与测试](models/sitstand_hd1910_3000/README.md) · [训练方法](docs/sitstand-official-scratch-20261021.md) |
| 拾取 | 基于 HD1910 重新训练，1001轮 / seed221149 | 用户实测拾取效果通过；仿真下探中位数约29mm | [模型与测试](models/pick_hd1910_1001/README.md) · [训练方法](docs/pick-seed-screening-20261022.md) |
| 翻滚 | 基于 HD1910 重新训练，1000轮 | 1.6秒翻滚接起身；统一交接版本用户反馈良好 | [模型与测试](models/roulade_hd1910_1000/README.md) · [训练方法](docs/pick-roll-official-scratch-20261021.md) |
| 左 / 右踢球 | 导入官方踢球模型 | 左右各一个模型 | [左脚](https://github.com/Ley-WL/microduck-observer/blob/feat/vision-camera-npu/models/kick_left.onnx) · [右脚](https://github.com/Ley-WL/microduck-observer/blob/feat/vision-camera-npu/models/kick_right.onnx) |

**2026-10-22 更新：** 57鸭子统一交接版本实测反馈良好；模型权重保持，配套名义交接仿真256/256通过，见[测试结论与脚本](docs/shared-walk-handoff-20261022.md)。

以上按当前Observer模型路由与最近部署记录整理；已部署不等于完整实物验收。后续训练提交仅在现役模型或其结论变化时更新本表，详细实验留在专题文档。

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

常用任务：`StandHold`（站立保持）、`train_walk_observer.py`（当前行走训练方法）、`GetupMechanical`（现行机械限位起身）、`SitStandContact`（坐站接触），完整 ID 以 `list-envs` 为准。任务各自的奖励、PPO、动作 EMA、限位和观测合同继续明确区分，不能把一种技能的动作滤波随意套到其他技能。通用 PPO/奖励算法保留，移除的是旧硬件默认参数和默认官方教师权重。

新训练默认使用本地 TensorBoard；实验名有 `hd1910` 标识。默认不下载官方教师，不自动推送权重。迁移学习必须显式选择兼容 checkpoint，旧权重不会因为更换配置自动成为新硬件训练结果。

## 验证和预览

```bash
uv run --with pytest pytest tests/test_hardware_profile.py tests/test_hd1910_stand_cfg.py tests/test_infer_policy_bam.py
uv run python scripts/infer_policy.py --walking /path/to/policy.onnx
```

CPU 推理入口也使用同一 HD1910 JSON；`--no-bam` 已拒绝，避免退回旧 XML PD。Windows 可使用现有 `scripts/preview_hd1910.py --policy ...` 预览带配套 metadata 的本地策略。预览必须匹配对应策略的动作合同，不能代替实机验收。

历史实验、原始权重和证据保留在 `artifacts/` 和历史记录中。上游说明见 [参考归档](docs/reference/upstream-readme.md)；旧 `robot/microduck`、XL330 台架和 XML-PD `sim/body_server.py` 仅作历史参考，不是当前训练入口，`duck-body` 命令不再发布。具体清理范围与验证见 [整理记录](docs/hardware-profile.md)。

代码沿用 Apache-2.0；模型/STL 的原始来源与许可声明继续保留，详见归档 README 和 HD1910 参数来源文档。
