"""Archive a completed finite experiment without promoting its weights."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
folder=root/'artifacts/sit-transfer-train-20261010'
evidence=folder/'server-train'
comparison=json.loads((evidence/'comparison.json').read_text(encoding='utf-8'))
gpu=json.loads((evidence/'gpu-robust/summary.json').read_text(encoding='utf-8'))
assert (evidence/'job.exit').read_text().strip()=='0' and (evidence/'completed.txt').is_file()
assert comparison['selected'] is None and not gpu['accepted']
entry='''> **2026-10-10 坐站过渡验证/1000轮最终结果（最新覆盖）：** 独立SitStandTransfer训练和CPU/GPU评估全部结束，job.exit=0/completed.txt存在、launcher已退出；训练26分33秒。新增切换物理俯仰扰动、一帧速度延迟和两膝2秒过程指标，修正CPU trial间BAM阻尼/摩擦残留。服务器严格同初态基线名义28/32、扰动9/16；100/500/999轮名义均28/32，扰动8/16、8/16、11/16。最终多余膝行程增加22.9%，丢失一个原成功例，过程筛选selected=null；GPU robust两seed循环17/32、13/32、16/32、20/32，保持25/32、29/32、29/32、28/32，未正式通过。归一化1000轮逐值保持、核心训练源哈希不变；3项过程门及先前20项检查通过。额外两组gyro一致性监督短实验亦因丢失成功例排除。BAM kd_scale为反电动势速度项，未辨识真实HD1910固件D20，不称实机PD已对齐或唯一根因。没有新部署/实物动作/自动续训，板上仍sit-smooth-v1候选、实物改善待独立验证。详见平级RL artifacts/sit-transfer-train-20261010/README.md与server-train/，本机早期基线不同运行环境不能混配服务器候选。

'''
paths=[root/'AGENTS.md',root/'docs/server-training.md',root/'docs/hd1910-stand-training.md',
       root.parent/'microduck-observer/AGENTS.md',
       root.parent/'microduck-observer/docs/实测记录/README.md',
       root.parent/'microduck-observer/docs/实测记录/03-舵机实测.md']
for path in paths:
    text=path.read_text(encoding='utf-8')
    if not text.startswith(entry):path.write_text(entry+text,encoding='utf-8')
readme=folder/'README.md'
text=readme.read_text(encoding='utf-8')
append='''
## 完整作业结束确认

训练及所有评估已结束：job.exit=0、completed.txt存在、launcher1662112已退出。最终GPU robust两seed循环为17/32、13/32、16/32、20/32；站保持25/32、29/32，坐保持29/32、28/32，accepted=false（成功率未达标且没有正式匹配过程参考/200例每组）。不自动续训/部署。GPU试验采用增益/反馈等robust配置，与CPU过渡物理扰动挑战分别报告，不混成同一种压力测试。

同999检查点两次导出的CPU candidate SHA783b3b80…与GPU policy SHA1e1fa9bc…不同；全部initializer逐值相同、3000仿真观测ONNX输出误差0，见export-equivalence.json。不能把CPU文件SHA写成实际GPU文件SHA。最终完整证据已压缩下载train-final.tar.gz并展开server-train/，保留日志/模型/元数据/逐trial结果/最终归一化检查。训练耗时26分33秒，不含导出和评估时间。
'''
if '## 完整作业结束确认' not in text:readme.write_text(text+append,encoding='utf-8')
print(json.dumps({'finished':True,'selected':None,'deployed':False,'recordsUpdated':len(paths)}))
