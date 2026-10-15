"""Simulation teacher query only; never a deployed command calibration."""
import json
from pathlib import Path


class GuidanceMap:
    def __init__(self,path):
        self.rows=json.loads(Path(path).read_text(encoding='utf-8'))['rows']
        assert [r['targetVx'] for r in self.rows]==[-.2,-.1,0.,.1,.2]
        assert all(r['targetYaw']==[-.6,0,.6] for r in self.rows)
        self.teacher_vx=[r['teacherVx'] if isinstance(r['teacherVx'],list) else [r['teacherVx']]*3 for r in self.rows]
        assert self.teacher_vx[2]==[0,0,0] and self.rows[2]['teacherYaw'][1]==0
        self.cache={}

    def query(self,observation):
        actor=observation['actor'].clone()
        assert actor.shape[-1]==61
        actor[:,48:51]=self.tensor(actor[:,48:51])
        return {**observation,'actor':actor}

    def tensor(self,command):
        import torch
        key=(command.device,command.dtype)
        if key not in self.cache:
            self.cache[key]=[torch.tensor(values,device=command.device,dtype=command.dtype) for values in
                ([-.2,-.1,0.,.1,.2],[-.6,0.,.6],self.teacher_vx,
                 [r['teacherYaw'] for r in self.rows])]
        grid_x,grid_y,source_x,source_y=self.cache[key]
        x=command[:,0].clamp(-.2,.2).contiguous();y=command[:,2].clamp(-.6,.6).contiguous()
        ix=(torch.searchsorted(grid_x,x,right=True)-1).clamp(0,3)
        iy=(torch.searchsorted(grid_y,y,right=True)-1).clamp(0,1)
        ax=(x-grid_x[ix])/(grid_x[ix+1]-grid_x[ix]);ay=(y-grid_y[iy])/(grid_y[iy+1]-grid_y[iy])
        output=command.clone()
        low_x=source_x[ix,iy]*(1-ay)+source_x[ix,iy+1]*ay
        high_x=source_x[ix+1,iy]*(1-ay)+source_x[ix+1,iy+1]*ay
        output[:,0]=low_x*(1-ax)+high_x*ax
        low=source_y[ix,iy]*(1-ay)+source_y[ix,iy+1]*ay
        high=source_y[ix+1,iy]*(1-ay)+source_y[ix+1,iy+1]*ay
        output[:,2]=low*(1-ax)+high*ax
        return output
