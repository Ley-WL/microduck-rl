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

## 唯一硬件入口
本仓库是基于 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) 的 HD1910 适配与训练实验分支。

十拍舞蹈训练任务、奖励、评估和续训脚本见 [训练记录](docs/dance10-training.md)。包含 Dance10、Dance10Expressive（大幅）及 Dance10Resilient（抗扰）任务。

已部署的 v5 推理权重与 metadata 位于 [microduck-observer/models](https://github.com/Ley-WL/microduck-observer/tree/main/models)（`dance10.onnx` / `dance10.metadata.json`），ONNX SHA256：`cd353fb203dd8edeb468476651280d9e03dcaabca53aef0fbfb91d9c66c8ee02`。完整 PPO checkpoint、训练日志及本地 `artifacts/` 未随代码发布；演示视频见上方；ONNX 不能替代续训 checkpoint。服务器编排脚本保留历史实验路径，运行前需配置自己的路径和种子 checkpoint。

v5 抗扰评估仍有失败与退化案例，公开与部署不代表实机稳定性验收。代码为 Apache-2.0；3D 模型沿用上游 CC BY-SA-NC 等原声明，不代表所有资源均可无限制商用。

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

# 新训练先做 64 环境、5 轮冒烟；通过后再按任务决定预算
uv run train Mjlab-Walk-Flat-MicroDuck-HD1910 --env.scene.num-envs 64 --agent.max-iterations 5
uv run train Mjlab-GetupMechanical-Flat-MicroDuck-HD1910 --env.scene.num-envs 64 --agent.max-iterations 5

# 导出保留内嵌观测归一化；明确指定本地 checkpoint
uv run python scripts/export.py Mjlab-Walk-Flat-MicroDuck-HD1910 --checkpoint-file /path/to/model.pt
```

常用任务：`StandHold`（站立保持）、`Walk`（行走）、`GetupMechanical`（现行机械限位起身）、`SitStandContact`（坐站接触），完整 ID 以 `list-envs` 为准。任务各自的奖励、PPO、动作 EMA、限位和观测合同继续明确区分，不能把一种技能的动作滤波随意套到其他技能。通用 PPO/奖励算法保留，移除的是旧硬件默认参数和默认官方教师权重。

新训练默认使用本地 TensorBoard；实验名有 `hd1910` 标识。默认不下载官方教师，不自动推送权重。迁移学习必须显式选择兼容 checkpoint，旧权重不会因为更换配置自动成为新硬件训练结果。

## 验证和预览

```bash
uv run --with pytest pytest tests/test_hardware_profile.py tests/test_hd1910_stand_cfg.py tests/test_infer_policy_bam.py
uv run python scripts/infer_policy.py --walking /path/to/policy.onnx
```

CPU 推理入口也使用同一 HD1910 JSON；`--no-bam` 已拒绝，避免退回旧 XML PD。Windows 可使用现有 `scripts/preview_hd1910.py --policy ...` 预览带配套 metadata 的本地策略。预览必须匹配对应策略的动作合同，不能代替实机验收。

历史实验、原始权重和证据保留在 `artifacts/` 和历史记录中。上游说明见 [参考归档](docs/reference/upstream-readme.md)；旧 `robot/microduck`、XL330 台架和 XML-PD `sim/body_server.py` 仅作历史参考，不是当前训练入口，`duck-body` 命令不再发布。具体清理范围与验证见 [整理记录](docs/hardware-profile.md)。

代码沿用 Apache-2.0；模型/STL 的原始来源与许可声明继续保留，详见归档 README 和 HD1910 参数来源文档。
