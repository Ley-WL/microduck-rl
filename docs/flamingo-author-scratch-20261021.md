# FlamingoCycleHard：公开配方从头训练

用户要求“与作者保持一致，然后训练看看”。本轮取消前两次自定义ONNX迁移，以固定上游commit `0bf989701f14d4008c866ffd6f09067870f99379` 注册的 `Mjlab-FlamingoCycleHard-Flat-MicroDuck` 公开环境/PPO配置从头训练。

## 证据边界与执行选择

上游设计文档介绍先研究单侧保持、再开发双侧Cycle；实际Cycle实现使用flag/side命令、混合初态和完整渐进课程。公开仓库没有找到发布run `flamingo-cycle-r2-hard-20260829-0245` 的完整启动命令或起始checkpoint记录，不能声称已精确恢复该run，也不臆造stage1→stage2权重加载链。本轮采用公开CycleHard任务支持的随机初始化训练入口，不额外创建阶段切换方案。

来源：

- [固定上游Cycle配置](https://github.com/pollen-robotics/microduck_rl/blob/0bf989701f14d4008c866ffd6f09067870f99379/src/mjlab_microduck/tasks/microduck_flamingo_cycle_env_cfg.py)
- [设计与实际实现说明](https://github.com/pollen-robotics/microduck_rl/blob/0bf989701f14d4008c866ffd6f09067870f99379/docs/superpowers/specs/2026-08-28-flamingo-design.md)

与作者公开配置一致：奖励及课程参数、MLP512/256/128 ELU、actor/critic在线归一化、初始探索std1、PPO学习率1e-3 adaptive、24步/环境、5epochs/4minibatches、4096环境、配置默认3000轮。采用同类 `MicroduckOnPolicyRunner`（继承VelocityOnPolicyRunner），一次连续learn，不分段重建环境、不人为回拨课程。

必要差异：使用当前replica几何/惯性、HD1910 P5与电压/延迟参数、Observer机械及动作限位；接触容量200/256；TensorBoard本地记录，禁用上传。seed211023为本轮明确选择，不能称作者同seed。现有mjlab1.3.0环境复用，未新增依赖；没有重新辨识执行器。

明确取消：ONNX加载、合成normalizer count、恢复一部分权重、新critic单独预热、手动设置成熟策略探索std、微调专用学习率。训练入口运行时断言actor/critic归一化count均为0、Adam状态为空、课程步数0。64环境5轮冒烟与正式训练独立初始化，不继承冒烟权重。

## 调度、评估和文件

训练入口：`scripts/server/flamingo_author_scratch.py`。复用经测试的 `src/mjlab_microduck/tasks/flamingo_hd1910_curriculum.py` 配置适配层。

独立服务器目录：`/CodeRepo/Code/leiwu30/hd1910-flamingo-author-scratch-20261021`。

现成Python：`/CodeRepo/Code/leiwu30/1_code/microduck-training/microduck-rl/.venv/bin/python`。

先跑4项配置/硬件测试，再通过原子launch.lock与GPU资源门启动。正式只调用一次learn(3000)，沿用save_interval100；原runner零基命名对应已完成501/1001/1501/2001/2501/3000次更新时导出并做96例仿真。仿真期间训练进程保留环境、优化器和RNG状态；不重置训练世界。评价入口与先前相同，每场初态hash必须匹配原版基线90/96。固定筛选集不当作独立实物验收。

进度见 `evidence/progress.json`，启动信息 `launch.json`，训练日志 `train.log`，对照汇总 `comparison.json`，最终退出码 `job.exit`。原模型只作为仿真基线，训练脚本没有policy/checkpoint加载参数。保留检查点，不自动部署、不修改机器人。

前一ONNX课程迁移已完成3000轮，最终保持0/96、未倒96/96、落脚96/96，拒绝采用；旧launcher1401494已退出。

## 启动核验

launcher2977838，GPU1。4项配置/硬件测试通过，64环境5轮从头训练冒烟通过，随后进入正式train阶段。新增入口Python AST、UTF-8无BOM/LF检查通过，未提交推送。当前无新一轮效果结论。

额外静态源核对：两份Flamingo配置排除import节点后AST与固定上游逐项相同；配置直接引用的38个MDP函数/类的AST均与固定上游相同。该检查不等于完整框架和全部递归依赖逐字一致，硬件相关差异仍以上述清单为准。

正式启动确认：日志已到零基iteration16/3000，运行合同确认fromScratch=true、无ONNX/ckpt载入、无critic预热、初始actor/critic归一化count和优化器状态均为0。GPU1采样总占用22189MiB、利用率68%（非本任务独占峰值）；单迭代约1.35秒，训练尚无效果验收结论。

## 约2530轮进度检查

launcher2977838仍运行，日志iteration2527/3000，剩余训练ETA约11分钟（不含末次评估）。已完成501/1001/1501/2001/2501轮评估，保持成功依次48/48/15/35/39，分母均96；未倒均96/96，落脚依次96/96/96/93/96。原模型基线保持90/96，当前仍未超过。

分侧结果：1001轮抬左脚三场均0/16，抬右脚三场均16/16；2501轮抬左脚仍三场0/16，抬右脚三场各13/16。2501轮无推扰右脚平均高度9.51cm，左脚几乎贴地。因此从头训练确实学会了右脚侧动作，但没有解决双侧不对称，不能把48/96称为两侧都学会。进入Hard阶段后成功数下降是时序关联，尚不能据此判定唯一原因。全部候选均不部署。

阶段证据已保存至 `artifacts/flamingo-author-scratch-20261021/`：comparison-through-2501.json、results-1001.json、results-2501.json。最终3000轮评估尚未结束。

## 最终完成：3000轮，单侧成功，不采用

最终progress=complete、job.exit=0，正式3000轮/72000环境步完成，全部阶段仿真结束。3000轮保持48/96、未倒96/96、落脚96/96；抬左脚三个场景均0/16，抬右脚三个场景均16/16。无推扰右脚平均抬高9.85cm，左脚基本贴地。相较原版90/96仍明显不足，不能当作合格双侧金鸡独立，不部署、不自动续训。

候选SHA256：`e01fa49f018b6ead35760b39df67102328cba439cfb985c1daf2ee9f213785f2`。本地 `artifacts/flamingo-author-scratch-20261021/final/` 保存完整阶段汇总comparison.json、最终results.json、policy.onnx及左右无推扰代表环境视频。完整checkpoint和训练日志仍保留服务器原目录。

## 后续用户授权：仅抬右脚部署

用户随后接受仅抬右脚，2026-10-21已部署192.168.137.240，服务runtime-20261021-flamingo-right-3000；板端校验通过，未发动作，实物测试待完成。上面的不采用/不部署是双侧评估后的历史决定。[正式模型、元数据及评估结果](../models/flamingo_right_hd1910_3000/README.md)。

复现需先用固定上游原版ONNX执行 `PYTHONPATH=src MUJOCO_GL=egl python scripts/server/flamingo_hd1910_trial.py eval --policy <原版policy.onnx> --output evidence/baseline --envs 16` 生成配对基线，再执行 `PYTHONPATH=src FLAMINGO_GPU=1 python scripts/server/flamingo_author_scratch.py launch`。使用独立干净工作目录，launch.lock防止重复启动；需现有Linux/CUDA训练环境。前两轮迁移脚本保留为历史对照，当前训练入口只有flamingo_author_scratch.py。
