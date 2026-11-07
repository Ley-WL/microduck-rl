# 最新配重轮滑 · 10000 更新

2026-11-07 用户确认保存本候选及训练方法。未部署到 Observer，不代表实机验收。

- 最终模型：`policy.onnx`，SHA256 `50282524d226c95bf5469140714d9499e55226f67ac74ee3b680209c6441f1d8`。
- 初始模型：`baseline.onnx`，SHA256 `539e6b2e7e4c7603ddd35ef27bc06742b292f9f3eede567f82111c0975375160`。
- 总质量 822.58644 g；4096 并行环境，seed 261106，10000 次 PPO 更新，训练耗时 03:58:45。
- 61 维观测、14 维动作，包含观测归一化；动作无 EMA。推力 ±0.6、航向误差 ±0.5 rad，不能当作线速度/角速度目标。
- 15 输入 × 16 环境名义测试：239/240 未倒；推力 +0.3、航向误差 +0.5 rad 有 1/16 失稳。小推力死区和直行偏转仍存在。不是 Observer 摇杆闭环验收，也未证明优于旧模型。

[训练方法](../../docs/swizzle-823g-20261106.md) · [10 秒原速仿真](preview.mp4) · [完整评估](evaluation.json)

`training-source.tar.gz` 是本次服务器实际使用的 src/ 和 scripts/ 源码快照，不含 Python 缓存；`source-manifest.json` 是训练时源码哈希记录。`environment.txt` 记录运行环境的包版本，硬件为 RTX 4090、Linux/CUDA。快照保留当时依赖行为，避免当前工作区其他实验改动影响复现。

## 复现

使用兼容 Linux/CUDA 环境，依照仓库 pyproject.toml/uv.lock 安装依赖，并核对 environment.txt；该文件是环境记录，不是跨平台锁文件。在新的空目录解包，勿覆盖正在训练的目录：

```bash
mkdir swizzle-reproduction
tar -xzf training-source.tar.gz -C swizzle-reproduction
cp baseline.onnx swizzle-reproduction/baseline.onnx
cd swizzle-reproduction
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
unset MICRODUCK_WARM_START
python scripts/server/train_swizzle_823g.py pipeline
```

入口先执行 64 环境 × 5 更新预检，再正式训练及最终评估。启动会生成 evidence/；不要重复执行覆盖结果。actor 和观测均值/方差由基线 ONNX 迁移，normalizer count=1000000、探索 std=0.1；critic、优化器、课程重新建立，不是完整 checkpoint 续训。GPU 数值和随机过程不保证逐位复现。

`preview.mp4` 是额外 8 环境、10 秒、25 fps 的名义回放，非原始 240 环境评估录像；`render-audit.json` 和 `video-results.json` 对应这次录像。
