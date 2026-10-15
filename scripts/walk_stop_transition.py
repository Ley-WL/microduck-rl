"""Find the first second of filtered command decay in complete offline worlds."""
import numpy as np

def stop_transition_mask(commands,steps=50):
    commands=np.asarray(commands)
    if commands.ndim!=3 or commands.shape[-1]!=3 or steps<1 or not np.isfinite(commands).all():
        raise ValueError('Expected finite [time,world,3] commands and positive window')
    magnitude=np.linalg.norm(commands,axis=-1)
    decay=np.zeros_like(magnitude,dtype=bool)
    decay[1:]=(magnitude[1:]<.9*magnitude[:-1])&(magnitude[:-1]>.001)
    start=decay.copy();start[1:]&=~decay[:-1]
    mask=np.zeros_like(start)
    for tick,world in zip(*np.nonzero(start)):
        mask[tick:min(tick+steps,len(mask)),world]=True
    return mask
