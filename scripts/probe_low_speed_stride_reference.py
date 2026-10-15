"""Offline-only pulse-stride feasibility probe; original desired commands retained."""
import sys,runpy,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'scripts'))
import walk_feedback_teacher
Base=walk_feedback_teacher.FeedbackTeacher
class PulseTeacher(Base):
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.debt=0.;self.walking=False;self.sign=0
    def query(self,command,actual,dt=.02,enabled=True):
        command=np.asarray(command,dtype=np.float32);speed=float(command[0])
        if not enabled or not .001<abs(speed)<.08:
            self.debt=0.;self.walking=False;self.sign=0
            return super().query(command,actual,dt,enabled)
        sign=np.sign(speed)
        if sign!=self.sign:self.debt=0.;self.walking=False;self.sign=sign
        self.debt+=dt*(speed-float(actual[0]))
        if sign*self.debt>.04:self.walking=True
        if sign*self.debt<-.04:self.walking=False
        query=command.copy();query[0]=sign*.18 if self.walking else 0.
        return super().query(query,actual,dt,enabled)
walk_feedback_teacher.FeedbackTeacher=PulseTeacher
runpy.run_path('scripts/evaluate_walk_runtime_cpu.py',run_name='__main__')
