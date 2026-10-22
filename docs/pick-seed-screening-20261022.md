# 拾取：官方配方多种子筛选

用户要求给出并执行解决方案，同时保持官方训练方法。2026-10-22 启动两组独立从头训练，尝试避免此前浅下探的收敛结果。换种子并不保证解决，结果必须由仿真确认。

| 种子 | GPU | pipeline PID | 服务器目录 | 首轮预算 |
| --- | --- | --- | --- | --- |
| 221141 | 1 | 295578 | `/CodeRepo/Code/leiwu30/hd1910-pick-seed221141-20261022` | 3001 更新 |
| 221149 | 2 | 295972 | `/CodeRepo/Code/leiwu30/hd1910-pick-seed221149-20261022` | 3001 更新 |

两组保留原官方任务、奖励、PPO、课程及 4096 环境，只保留已有 HD1910 / Observer 限位适配。官方配置的最大轮数仍是 20000；3001 是本次筛选停止点，课程不压缩。没有载入旧模型或合成归一化计数。原 20000 轮目录和实机模型不变。

启动器新增 `OFFICIAL_SEED`、`OFFICIAL_SCREEN_UPDATES`，未设置时维持旧行为。每组约 1000、3000 轮评估同初态 32 例；`depth.json` 保存首 2.8 秒最低嘴高的中位数和三个回站窗口的稳定帧比例。原 success 指标保留，但它是本地 25 mm / 三周期判据，不是官方成功率。

筛选看下探与回站同时改善：现用权重约 18 mm、官方权重在 HD1910 条件约 35.5 mm、旧新训 20000 轮约 55.5 mm。先争取优于旧新训并接近官方权重，再以现用模型作为实际目标。3001 轮结果并非最终收敛证明；本轮不自动续训或部署，保存完整检查点供评审后续训。候选需进一步检查动作裁剪及官方 2.8 秒切换回行走的表现。

已验证：两组官方配置测试各 6 项通过；新增启动参数测试 5 项通过。启动时进入 smoke 阶段，正式进度以各自 `evidence/progress.json`、`train.log`、`job.exit` 为准，禁止重复启动。

复用服务器既有 `.venv`，未安装依赖。训练数据、日志、环境均留在独立服务器目录，不加入仓库；未提交推送。测试命令：`python -m pytest tests/test_official_screening_launcher.py tests/test_pick_roll_observer_official.py -q`（后者需现有训练环境）。

此前诊断和完整对照见 [拾取/翻滚训练记录](pick-roll-official-scratch-20261021.md)。

## 192.168.137.233 候选部署

用户于 2026-10-22 明确要求部署刚查看的种子 221149 / 1001 更新拾取模型。通过 Observer model-registry 导入并选用，ID user-9e0e90fd-39c4-4f60-8462-3127977d7d33，SHA256 b9f88799be0f979115f8d22fefbc3d4db8c95e4567f6ac04f42c6203cece5141。导入和选择均返回 saved=true，revision=2，下次拾取启动生效。原 pick-default 保留为 previous，可回滚；其他槽位选择不变。使用兼容 kissqy_ground_pick 运行合同（只是运行合同名称，不是权重来源），保留当前 P5/D20、2.8秒窗口、50Hz、raw history/no EMA。未重启服务或发动作。候选尚未实机验收；板端记录 /home/radxa/pick1001-deployment-20261022/。

## 复现当前模型的方法

在独立Linux/CUDA仓库副本中复用项目锁定环境。对照文件 `baseline.onnx` 应使用旧拾取权重（SHA256 `2b69fa681ed06f7ded9e2a2d12ba5c6776ec87308ac4a299e4b8196199c17f24`，Observer提交c842350的models/xgoduck_pick.onnx），仅供评估，不载入训练。

```bash
uv sync
# 将上述对照权重复制至仓库根目录 baseline.onnx；确认GPU空闲。
OFFICIAL_SKILL=pick OFFICIAL_SEED=221149 OFFICIAL_SCREEN_UPDATES=3001 OFFICIAL_GPU=2   uv run python scripts/server/train_pick_roll_official.py launch
```

入口自动执行配置测试、64环境5更新试跑、32例对照、4096环境训练及1001/3001评估。课程时间不压缩；省略筛选参数时官方pick预算为20000。每次运行须使用独立目录，launch.lock防止重复启动。输出 `evidence/model1001/policy.onnx` 即本次发布版本所在检查点；随机种子相同也不承诺跨硬件逐位复现。

2026-10-22 用户确认233实机效果并要求提交，选用1001更新而非3001更新；发布模型与简表见 [README](../models/pick_hd1910_1001/README.md)。训练配置测试6项、启动参数测试5项通过。原始日志/视频/NPZ不入库。
