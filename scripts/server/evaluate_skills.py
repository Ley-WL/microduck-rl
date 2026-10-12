"""Held-out ONNX skill trials. No timeout resets; latch every other reset as failure.

Getup: six fallen cohorts, random yaw/joint/tilt perturbations, 15 seconds,
final 3 seconds >=95% upright and quiet. Sitstand: two full repeated-cycle
scenarios plus both holds, all dwell endpoints pass and no excessive tipping.
Acceptance requires >=95% success for EVERY scenario and seed, not just pooled.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import random
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.microduck_hd1910_skills_env_cfg import (
    POSE_NAMES, SIT_Z, STAND_Z, SITTING_TARGET_OVERRIDES,
)


def stability_gates(tilt, height, xy_speed, angular_speed, leg_error, sit, heights=(STAND_Z,SIT_Z)):
    target = heights[int(sit)]
    return {"tilt": tilt < 20., "height": np.abs(height-target) < .012,
            "horizontalSpeed": xy_speed < .2, "angularSpeed": angular_speed < 2., "legPose": leg_error < .25}


def stable_mask(tilt, height, xy_speed, angular_speed, leg_error, sit, heights=(STAND_Z,SIT_Z)):
    return np.logical_and.reduce(list(stability_gates(tilt, height, xy_speed, angular_speed, leg_error, sit, heights).values()))


def scope_complete(results, scenarios, seeds):
    expected = {(scenario, seed) for scenario in scenarios for seed in seeds}
    actual = [(result['scenario'], result['seed']) for result in results]
    return len(actual) == len(expected) and set(actual) == expected


def compare_process_results(results, reference):
    """A sitstand endpoint pass alone cannot qualify transition quality."""
    if reference is None:
        return {'passed':False,'reason':'No matching process reference supplied'}
    original={(r['scenario'],r['seed']):r for r in reference}
    current={(r['scenario'],r['seed']):r for r in results}
    if set(original)!=set(current) or len(original)!=len(reference) or len(current)!=len(results):
        return {'passed':False,'reason':'Process reference scope differs'}
    edges=[];baseline=[];lost=0
    for key,row in current.items():
        old=original[key]
        if not row.get('initialStateDigest') or row['initialStateDigest']!=old.get('initialStateDigest'):
            return {'passed':False,'reason':'Initial state/actuator randomization is missing or unmatched'}
        if (row['envs']!=old['envs'] or row.get('profile')!=old.get('profile')
            or any(len(r.get(field,[]))!=r['envs'] for r in (row,old)
                   for field in ('successByTrial','transitionSmoothnessByTrial'))):
            return {'passed':False,'reason':'Missing or incompatible per-trial process measurements'}
        if any([t['time'] for t in trial if t['time']>0]!=[t['time'] for t in prior if t['time']>0]
               for trial,prior in zip(row['transitionSmoothnessByTrial'],old['transitionSmoothnessByTrial'])):
            return {'passed':False,'reason':'Per-trial transition windows differ'}
        lost+=sum(a and not b for a,b in zip(old['successByTrial'],row['successByTrial']))
        edges.extend(t for trial in row['transitionSmoothnessByTrial'] for t in trial if t['time']>0)
        baseline.extend(t for trial in old['transitionSmoothnessByTrial'] for t in trial if t['time']>0)
    if not edges or len(edges)!=len(baseline):return {'passed':False,'reason':'No complete matched transitions'}
    travel=np.mean([r['excessTravelDeg'] for r in edges],axis=0)
    before=np.mean([r['excessTravelDeg'] for r in baseline],axis=0)
    frames=np.percentile([r['maxFrameChangeDeg'] for r in edges],95,axis=0)
    old_frames=np.percentile([r['maxFrameChangeDeg'] for r in baseline],95,axis=0)
    passed=lost==0 and bool(np.all(travel<=before*1.05+1e-6) and np.all(frames<=old_frames*1.05+1e-6))
    return {'passed':passed,'lostSuccessfulTrials':lost,'meanExcessKneeTravelDeg':travel.tolist(),
            'referenceMeanExcessKneeTravelDeg':before.tolist(),'p95MaxKneeFrameChangeDeg':frames.tolist(),
            'referenceP95MaxKneeFrameChangeDeg':old_frames.tolist(),'tolerance':1.05,
            'scope':'Relative process regression gate; 5% diagnostic tolerance, not a servo stop threshold'}


def verify_assembled_skill_contacts(model, masks=(1,0)):
    contacts = [i for i in range(model.ngeom)
                if model.geom(i).name.startswith('robot/body_ground_')]
    if len(contacts) != 68:
        raise ValueError('Expected 68 assembled body ground contact meshes')
    if any((model.geom_contype[i], model.geom_conaffinity[i]) != masks
           or model.geom_condim[i] != 3 for i in contacts):
        raise ValueError('Assembled body contact contract is disabled or changed')
    terrain = [i for i in range(model.ngeom) if model.geom(i).name == 'terrain']
    if not terrain or not any(model.geom_conaffinity[i] & 1 for i in terrain):
        raise ValueError('Terrain cannot accept body ground contacts')


def evaluate(a, scenario, seed):
    task = f"Mjlab-{'Getup' if a.kind == 'getup' else 'SitStand'}-Flat-MicroDuck-HD1910"
    meta = json.loads(a.onnx.with_suffix('.metadata.json').read_text())
    contact_candidate = meta.get('skillPhysicsRevision') == 'sit-self-contact-bounded-target-v3'
    if contact_candidate:
        if a.kind != 'sitstand' or (meta['task'] not in ('Mjlab-SitStandContact-Flat-MicroDuck-HD1910','Mjlab-SitStandTransfer-Flat-MicroDuck-HD1910') and not meta['task'].startswith('Mjlab-SitStandProcess')):
            raise ValueError('Candidate task/skill mismatch')
        task = meta['task']
    cfg = load_env_cfg(task, play=a.profile == "nominal")
    heights = (cfg.commands['twist'].stand_z,cfg.commands['twist'].sit_z) if a.kind == 'sitstand' else (STAND_Z,SIT_Z)
    cfg.curriculum.clear()
    cfg.scene.num_envs = a.envs
    cfg.episode_length_s = 10000.
    cfg.terminations.pop("time_out", None)
    cohort = POSE_NAMES.index(scenario) if a.kind == "getup" else (1 if scenario.startswith("sit_") else 0)
    cfg.events["set_ground_state"].params["pose_probabilities"] = tuple(float(i == cohort) for i in range(8))
    for command in cfg.commands.values():
        if hasattr(command, "resampling_time_range"):
            command.resampling_time_range = (10000., 10000.)
    if a.profile == "robust":
        for name, magnitude in (("randomize_com", .015), ("randomize_head_com", .01)):
            if name in cfg.events:
                cfg.events[name].params["ranges"] = (-magnitude, magnitude)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(a.onnx), sess_options=options, providers=["CPUExecutionProvider"])
    # Actuator supply voltage and sag are startup-randomized, before reset.
    # Seed the constructor too; reset(seed=...) alone cannot pair these.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    cfg.seed=seed
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    writer = renderer = None
    try:
        verify_assembled_skill_contacts(env.sim.mj_model, (3,2) if contact_candidate else (1,0))
        obs, _ = env.reset(seed=seed)
        assert obs["actor"].shape == (a.envs, 61)
        robot = env.scene["robot"]
        initial_hash=hashlib.sha256()
        initial_tensors=[env.sim.data.qpos,env.sim.data.qvel,obs['actor']]
        for actuator in robot.actuators:
            for name in ('vin_tensor','vin_drop_gain','kp_scale','kd_scale'):
                tensor=getattr(actuator,name,None)
                if tensor is not None:initial_tensors.append(tensor)
        for tensor in initial_tensors:initial_hash.update(tensor.detach().cpu().numpy().tobytes())
        initial_digest=initial_hash.hexdigest()
        legs = [i for i, name in enumerate(robot.joint_names) if name.startswith(("left_", "right_"))]
        home = robot.data.default_joint_pos.clone()
        sitting = home.clone()
        for i, value in cfg.events['set_ground_state'].params['sit_overrides'].items():
            sitting[:, i] = value
        twist = env.command_manager.get_term("twist")
        failed = np.zeros(a.envs, dtype=bool)
        finite = True
        if a.video:
            import imageio.v2 as imageio
            renderer = mujoco.Renderer(env.sim.mj_model, height=480, width=640)
            data = mujoco.MjData(env.sim.mj_model)
            camera = mujoco.MjvCamera()
            camera.azimuth, camera.elevation, camera.distance = 150, -18, .7
            writer = imageio.get_writer(str(a.output / f"{a.kind}-{scenario}-{seed}.mp4"), fps=25)
        cycle = scenario.endswith("cycle")
        initial_sit = scenario.startswith("sit_")
        seconds = 20. if cycle else 15.
        # Two completed direction changes plus a repeated transition, all with hold.
        segments = [(2., initial_sit), (8., not initial_sit), (14., initial_sit), (20., not initial_sit)] if cycle else [(seconds, initial_sit)]
        if a.kind == "getup":
            segments = [(15., False)]
        samples = []
        gate_samples = []
        trace = []
        maximum_tilt = np.zeros(a.envs)
        process_frames=[]
        for tick in range(round(seconds / env.step_dt)):
            time = tick * env.step_dt
            sit = next(flag for end, flag in segments if time < end - 1e-6)
            twist.vel_command_b[:] = 0.
            if a.kind == "sitstand":
                twist.vel_command_b[:, 0] = float(sit)
            # Update only the known 61D contract command slice. Calling the
            # observation manager twice would advance sensor delay histories.
            obs["actor"][:, 48:51] = twist.vel_command_b
            actor_obs = obs["actor"].cpu().numpy()
            if session.get_inputs()[0].shape[0] == 1:
                values = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None]})[0] for row in actor_obs])
            else:
                values = session.run(None, {session.get_inputs()[0].name: actor_obs})[0]
            if not np.isfinite(values).all():
                raise ValueError("Non-finite ONNX actions")
            obs, reward, terminated, truncated, _ = env.step(torch.from_numpy(values).to(env.device))
            if a.kind=='sitstand':
                process_frames.append((time,bool(sit),env.action_manager.get_term('joint_pos').position_target.detach().cpu().numpy().copy()))
            failed |= (terminated | truncated).cpu().numpy()
            finite &= bool(torch.isfinite(obs["actor"]).all()) and bool(torch.isfinite(reward).all())
            tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1., 1.))).cpu().numpy()
            height = (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).cpu().numpy()
            xy = robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1).cpu().numpy()
            angular = robot.data.root_link_ang_vel_b.norm(dim=-1).cpu().numpy()
            target = sitting if sit else home
            error = (robot.data.joint_pos[:, legs] - target[:, legs]).abs().mean(dim=-1).cpu().numpy()
            maximum_tilt = np.maximum(maximum_tilt, tilt)
            samples.append(stable_mask(tilt, height, xy, angular, error, sit, heights))
            gate_samples.append(stability_gates(tilt, height, xy, angular, error, sit, heights))
            trace.append({"time": time, "sit": sit, "tilt0": float(tilt[0]), "height0": float(height[0]), "stableCount": int(samples[-1].sum())})
            if writer and tick % 2 == 0:
                data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
                mujoco.mj_forward(env.sim.mj_model, data)
                camera.lookat[:] = data.xpos[env.sim.mj_model.body("robot/trunk_base").id]
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
        samples = np.array(samples)
        success = ~failed
        dwell = []
        for end, sit in (segments[1:] if cycle else segments):
            window = 3. if a.kind == "getup" else 2.
            fraction = samples[round((end-window)/env.step_dt):round(end/env.step_dt)].mean(axis=0)
            success &= fraction >= .95
            begin_tick, end_tick = round((end-window)/env.step_dt), round(end/env.step_dt)
            gate_fractions = {name: np.array([sample[name] for sample in gate_samples[begin_tick:end_tick]]).mean(axis=0)
                              for name in gate_samples[0]}
            dwell.append({"endSeconds": end, "sit": sit, "stableFractions": fraction.tolist(),
                          "perGateStableFractions": {name: values.tolist() for name, values in gate_fractions.items()},
                          "failedTrialsByGate": {name: int((values < .95).sum()) for name, values in gate_fractions.items()}})
        if a.kind == "sitstand":
            success &= maximum_tilt < 45.
        if not finite:
            success[:] = False
        result = {"scenario": scenario, "seed": seed, "profile": a.profile, "envs": a.envs,
                  "initialStateDigest":initial_digest,
                  "successes": int(success.sum()), "successRate": float(success.mean()),
                  "successByTrial": success.tolist(), "resetOrTerminationByTrial": failed.tolist(),
                  "maxTiltByTrial": maximum_tilt.tolist(), "finite": finite, "dwell": dwell}
        result["finalState"] = {"tiltDeg": tilt.tolist(), "heightM": height.tolist(),
                                "speedXY": xy.tolist(), "angularSpeed": angular.tolist(), "legPoseErrorRad": error.tolist()}
        if process_frames:
            import importlib.util
            spec=importlib.util.spec_from_file_location('process_cpu',Path(__file__).parents[1]/'evaluate_skills_cpu.py')
            process_cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(process_cpu)
            result['transitionSmoothnessByTrial']=[process_cpu.transition_smoothness(
                [{'time':t,'sit':s,'target':target[i].tolist()} for t,s,target in process_frames],robot.joint_names)
                for i in range(a.envs)]
        (a.output / f"{a.kind}-{scenario}-{seed}.json").write_text(json.dumps({"result": result, "trace": trace}, indent=2))
        print(json.dumps({k: result[k] for k in ("scenario", "seed", "successRate", "finite")}), flush=True)
        return result
    finally:
        if writer:
            writer.close()
            renderer.close()
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["getup", "sitstand"], required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--envs", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[1009, 2027])
    p.add_argument("--profile", choices=["nominal", "robust"], default="robust")
    p.add_argument("--scenario", nargs="+")
    p.add_argument("--video", action="store_true")
    p.add_argument('--process-reference',type=Path,help='Matching GPU summary with per-trial transition metrics; required for sitstand acceptance')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    scenarios = list(POSE_NAMES[2:]) if a.kind == "getup" else ["stand_cycle", "sit_cycle", "stand_hold", "sit_hold"]
    selected = a.scenario or scenarios
    if not set(selected).issubset(scenarios):
        p.error("Unknown scenario")
    if len(set(selected)) != len(selected) or len(set(a.seeds)) != len(a.seeds):
        p.error("Scenarios and seeds must be unique")
    results = []
    reference=json.loads(a.process_reference.read_text())['results'] if a.process_reference else None
    for seed in a.seeds:
        for scenario in selected:
            results.append(evaluate(a, scenario, seed))
            process=compare_process_results(results,reference) if a.kind=='sitstand' else {'passed':True}
            summary = {"kind": a.kind, "policySha256": hashlib.sha256(a.onnx.read_bytes()).hexdigest(),
                       "profile": a.profile, "results": results,
                       "scopeComplete": scope_complete(results, scenarios, a.seeds),
                       "acceptanceSampleMinimum": 200,
                       'processComparison':process,
                       "accepted": process['passed'] and scope_complete(results, scenarios, a.seeds) and len(set(a.seeds)) >= 2 and a.envs >= 200 and all(r["successRate"] >= .95 and r["finite"] for r in results)}
            (a.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
