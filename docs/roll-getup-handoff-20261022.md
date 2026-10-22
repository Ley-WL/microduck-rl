# 翻滚后收腿接起身验证

本次只修改Observer切换流程，没有重训或更换模型：翻滚2秒→主动收腿→IMU躺倒姿态门→倒地起身→行走。

- 固定本轮0.10秒actor输出作为收腿目标；0.20/0.30秒探索候选未采用。
- 筛选32/32；锁定后新seed221227/221231各32环境，64/64完成并末1秒低运动。起身接管2.24～2.82秒，回行走5.24～6.46秒。
- 7.4V、HD1910 BAM、Observer限位、翻滚接触模型；起身滤波/EMA及raw action历史复现，未建模真实通信时延和实机执行器差异，无外扰，无物理随机化。仅名义仿真，非实机验收。
- 233已发布runtime-20261022-144051，未发动作，实测待用户。

[完整状态机与部署记录](https://github.com/Ley-WL/microduck-observer/blob/feat/vision-camera-npu/docs/roll-tuck-getup-20261022.md) · [64例结果](evidence/roll-getup-20261022/results.json)

历史2秒方案已被[统一交接版本](shared-walk-handoff-20261022.md)取代；下列脚本现为最终1.6秒版本。

当前复现入口：`scripts/server/evaluate_roll_getup.py --policy <roll.onnx> --getup <getup.onnx> --walk <walk.onnx> --output <directory>`。复用既有训练环境，无训练优化、硬件访问或新增依赖。
