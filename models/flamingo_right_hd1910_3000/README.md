# HD1910 右脚金鸡独立 3000 轮

采用作者公开 CycleHard 配方从头训练，适配 HD1910 P5、当前几何与 Observer 限位。[训练方法和历史评估](../../docs/flamingo-author-scratch-20261021.md)。

## 测试结论

最终 96 例：右脚保持 48/48，左脚保持 0/48；未倒和落脚均 96/96。原版双侧保持 90/96。本模型仅用于右脚动作，不能称双侧成功或实物验收通过。[最终结果](results.json) · [阶段对比](comparison.json)。

2026-10-21 用户授权部署到 Observer，服务固定左支撑、右抬脚。板端推理校验通过，实物动作待用户测试。hold=[1,-1,0]，lower=[0,-1,0]；61→14、50Hz、scale1、无EMA、无额外电压缩放、Observer有效限位。保留当前5000轮行走模型。

[ONNX](policy.onnx) · [运行元数据](policy.metadata.json)

SHA256：`e01fa49f018b6ead35760b39df67102328cba439cfb985c1daf2ee9f213785f2`。ONNX用于推理，不能替代完整PPO续训checkpoint；完整checkpoint与日志保留训练服务器，未入库。
