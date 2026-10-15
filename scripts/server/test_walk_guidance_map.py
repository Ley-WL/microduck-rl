import sys
from pathlib import Path
import torch
from walk_guidance_map import GuidanceMap
path=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[2]/'inputs/guidance-map.json'
g=GuidanceMap(path)
commands=torch.tensor([[r['targetVx'],0.,y] for r in g.rows for y in r['targetYaw']])
expected=torch.tensor([[x,0.,y] for row,xs in zip(g.rows,g.teacher_vx) for x,y in zip(xs,row['teacherYaw'])])
torch.testing.assert_close(g.tensor(commands),expected)
torch.testing.assert_close(g.tensor(torch.zeros(1,3)),torch.zeros(1,3))
queries=torch.randn(512,3)*.05
assert torch.isfinite(g.tensor(queries)).all()
q=torch.tensor([[.1-1e-6,0.,.2],[.1+1e-6,0.,.2]])
assert (g.tensor(q)[0]-g.tensor(q)[1]).abs().max()<1e-4
observation={'actor':torch.zeros(3,61),'critic':torch.ones(3,76)}
observation['actor'][:,48]=.2
original=observation['actor'].clone();query=g.query(observation)
torch.testing.assert_close(observation['actor'],original)
torch.testing.assert_close(query['actor'][:,48],torch.full((3,),g.teacher_vx[-1][1]))
assert query['critic'] is observation['critic']
torch.testing.assert_close(query['actor'][:,:48],original[:,:48])
print('guidance endpoints/idle/continuity/finite passed; no runtime mapping installed')
