from pathlib import Path
p=Path(__file__).with_name('diagnose_walk_runtime_ema.py').resolve()
s=p.read_text(encoding='utf-8').split('results=[]')[0]
exec(compile(s,str(p),'exec'))
import imageio.v2 as imageio
ns['bam'].actuator.kp=6;ns['action_alpha']=.45;ns['filtered_action']=np.zeros(14,dtype=np.float32);ns['filtered_gyro']=np.zeros(3);ns['reset']()
renderer=ns['mujoco'].Renderer(ns['model'],height=480,width=640)
cam=ns['mujoco'].MjvCamera();cam.distance=.65;cam.azimuth=135;cam.elevation=-18
video=out/'walk-v6-kp6-ema045.mp4'
with imageio.get_writer(video,fps=25) as writer:
 for tick in range(1000):
  ns['step']()
  if tick%2==0:
   cam.lookat[:]=ns['data'].qpos[:3];renderer.update_scene(ns['data'],camera=cam);writer.append_data(renderer.render())
renderer.close();print(str(video))
