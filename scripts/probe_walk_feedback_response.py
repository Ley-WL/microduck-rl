"""Offline teacher response probe; not a robot controller or trained student."""
import argparse
import runpy
import sys
from pathlib import Path
import walk_feedback_teacher

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--response-seconds',type=float,default=.35)
p.add_argument('--vx-kp',type=float,default=.3)
p.add_argument('--vx-ki',type=float,default=.8)
p.add_argument('--reverse-vx-kp',type=float)
p.add_argument('--reverse-vx-ki',type=float)
a,remaining=p.parse_known_args()
Base=walk_feedback_teacher.FeedbackTeacher

class ProbeTeacher(Base):
    def __init__(self,**kwargs):
        super().__init__(**kwargs,response_seconds=a.response_seconds,vx_kp=a.vx_kp,vx_ki=a.vx_ki)
    def query(self,command,actual,**kwargs):
        self.vx_kp=a.reverse_vx_kp if command[0]<0 and a.reverse_vx_kp is not None else a.vx_kp
        self.vx_ki=a.reverse_vx_ki if command[0]<0 and a.reverse_vx_ki is not None else a.vx_ki
        if not 0<=self.vx_kp<=2 or not 0<self.vx_ki<=2:
            raise ValueError('Invalid reverse response')
        return super().query(command,actual,**kwargs)

walk_feedback_teacher.FeedbackTeacher=ProbeTeacher
sys.argv=[sys.argv[0],*remaining]
runpy.run_path(str(Path(__file__).with_name('evaluate_walk_runtime_cpu.py')),run_name='__main__')
