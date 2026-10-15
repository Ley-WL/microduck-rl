"""CPU reward tests without importing the GPU simulator stack."""
import ast
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
source=ast.parse((root/'src/mjlab_microduck/tasks/mdp.py').read_text(encoding='utf-8'))
names={'hd1910_stride_quality','hd1910_stride_component_quality'}
ns={'torch':torch}
exec(compile(ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'stride','exec'),ns)
f=ns['hd1910_stride_component_quality']
p=torch.tensor([.0275,.04,.06]);z=torch.zeros_like(p);speed=torch.full_like(p,.05)
old=f(p,z,z+.018,z+.18,speed,cycle_time=.55,target_air=.18,target_height=.018)
new=f(p,z,z+.018,z+.18,speed,cycle_time=.55,target_air=.18,target_height=.018,min_stride=.06)
assert torch.allclose(old,torch.ones_like(old))
assert new[0]<new[1]<new[2] and new[2]==1
assert not f(p,z,z+.018,z+.18,z,min_stride=.06).any()
assert not f(-p,z,z+.018,z+.18,speed,min_stride=.06).any()
print('Low-speed distance gradient, idle/backward exclusion passed')
