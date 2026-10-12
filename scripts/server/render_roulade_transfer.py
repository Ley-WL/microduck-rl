"""Render cached paired states only; no new rollout/optimizer/hardware action."""
import os,json,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence/roulade-transfer'
free,util=map(int,subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free,utilization.gpu',
    '--format=csv,noheader,nounits'],text=True).strip().split(','))
assert free>12288 and util<30,'Do not compete for GPU rendering capacity'
os.environ.update(MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='1',MICRODUCK_ROULADE_INPUTS=str(ROOT/'inputs'))
import mujoco
from PIL import Image,ImageDraw
from mjlab_microduck.tasks.microduck_roulade_mechanical_env_cfg import get_roll_spec
spec=get_roll_spec()
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=[2.,2.,.01],rgba=[.72,.76,.8,1.])
model=spec.compile();data=mujoco.MjData(model)
baseline=np.load(OUT/'baseline-screen-v3/v6.4-seed90173.npz')
repair=np.load(OUT/'repair-screen-v3/v6.4-seed90173.npz')
def success(f):
    stable=(f['tilt']<20)&(f['height']>.095)&(f['accum']>=np.radians(330))&f['head']
    return stable[-15:].mean(0)>=.95
lost=np.flatnonzero(success(baseline)&~success(repair));world=int(lost[0]) if len(lost) else 0
ticks=[0,12,25,40,50,70,149];width=320;height=260
canvas=Image.new('RGB',(width*len(ticks),height*2),'white');draw=ImageDraw.Draw(canvas)
renderer=mujoco.Renderer(model,height=240,width=width)
camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.distance=.7;camera.azimuth=90;camera.elevation=-10
try:
    for row,(name,f) in enumerate([('Baseline',baseline),('Repair20',repair)]):
        for col,tick in enumerate(ticks):
            q=f['qpos'][tick,world];assert len(q)==model.nq
            data.qpos[:]=q;mujoco.mj_forward(model,data)
            camera.lookat[:]=[data.qpos[0],data.qpos[1],.10]
            renderer.update_scene(data,camera=camera)
            canvas.paste(Image.fromarray(renderer.render()),(col*width,row*height+20))
            draw.text((col*width+4,row*height+4),f'{name} {(tick+1)*.02:.2f}s tilt{f["tilt"][tick,world]:.1f}',fill='black')
finally:renderer.close()
canvas.save(OUT/'paired-roll-contact-sheet.png')
(OUT/'render-audit.json').write_text(json.dumps(dict(cachedStatesOnly=True,voltage=6.4,seed=90173,world=world,
    baselineSuccess=bool(success(baseline)[world]),repairSuccess=bool(success(repair)[world]),hardwareAccess=False,training=False),indent=2))
print('rendered cached paired world',world)
