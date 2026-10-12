"""Landing-based gait measurements, excluding brief contact chatter."""
import numpy as np


def summarize_heading_path(positions,headings,dt,target_yaw_rate=0.):
    positions=np.asarray(positions);headings=np.unwrap(np.asarray(headings),axis=0)
    delta=positions[:,:,:2]-positions[0,:,:2]
    forward=np.stack((np.cos(headings[0]),np.sin(headings[0])),axis=-1)
    side=np.stack((-forward[:,1],forward[:,0]),axis=-1)
    along=(delta*forward[None]).sum(-1);cross=(delta*side[None]).sum(-1)
    error=headings-headings[0]-np.arange(len(headings))[:,None]*dt*target_yaw_rate
    return {'headingDriftP95Deg':float(np.percentile(np.abs(np.rad2deg(error)),95)),
            'finalHeadingDriftP95Deg':float(np.percentile(np.abs(np.rad2deg(error[-1])),95)),
            'crossTrackP95M':float(np.percentile(np.abs(cross),95)),
            'projectedForwardTravelMeanM':float(np.mean(along[-1]))}


def summarize_gait(positions, contacts, air_times, heights, dt):
    positions,contacts,air_times,heights=map(np.asarray,(positions,contacts,air_times,heights))
    last={};strides=[];durations=[];peaks=[];raw=0
    for t in range(1,len(contacts)):
        for world,foot in np.argwhere(contacts[t]&~contacts[t-1]):
            raw+=1
            duration=float(air_times[t,world,foot])
            if duration<.06:
                continue
            durations.append(duration)
            start=max(0,t-round(duration/dt))
            peaks.append(float(np.max(heights[start:t+1,world,foot])))
            key=(int(world),int(foot))
            if key in last:
                previous=last[key]
                strides.append(float(np.linalg.norm(positions[t,world,foot,:2]-positions[previous,world,foot,:2])))
            last[key]=t
    return {'validLandings':len(durations),'rawLandings':raw,
            'contactChatterFraction':1-len(durations)/max(raw,1),
            'validStepCadenceHz':len(durations)/max(len(contacts)*dt*contacts.shape[1],1e-9),
            'medianStrideM':float(np.median(strides)) if strides else 0.,
            'medianLandingAirTimeSec':float(np.median(durations)) if durations else 0.,
            'medianSwingPeakM':float(np.median(peaks)) if peaks else 0.}
