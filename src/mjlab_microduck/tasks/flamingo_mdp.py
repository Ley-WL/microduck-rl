"""Flamingo helpers from pollen-robotics/microduck_rl 0bf989701f14d4008c866ffd6f09067870f99379 (Apache-2.0)."""
from .mdp import *
from .mdp import _DEFAULT_ASSET_CFG, _servo_joint_ids, _servo_default_joint_pos, _servo_joint_pos, _servo_joint_vel, _dataclass
def set_flamingo_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    joint_pose: list,
    base_roll: float,
    base_pitch: float,
    z_min: float,
    z_max: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    tilt_noise: float = 0.0,
    joint_noise_std: float = 0.0,
    standing_prob: float = 0.0,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
):
    """Spawn IN the one-foot pose (trunk rolled/pitched so the stance sole is flat),
    with joint + tilt noise and a random yaw; optionally a ``standing_prob`` bucket
    at HOME/upright for the stage-2 transition training. ``joint_pose`` is the
    14-servo pose (0-4 left leg, 5-8 neck/head, 9-13 right leg). qvel is zeroed."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)
    dev = env.device

    is_stand = torch.rand(num, device=dev) < standing_prob
    yaw = torch.rand(num, device=dev) * 2 * np.pi - np.pi
    roll = torch.full((num,), float(base_roll), device=dev)
    pitch = torch.full((num,), float(base_pitch), device=dev)
    if tilt_noise > 0.0:
        roll = roll + (torch.rand(num, device=dev) * 2 - 1) * tilt_noise
        pitch = pitch + (torch.rand(num, device=dev) * 2 - 1) * tilt_noise
    roll = torch.where(is_stand, torch.zeros_like(roll), roll)
    pitch = torch.where(is_stand, torch.zeros_like(pitch), pitch)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    # ZYX intrinsic Euler (yaw * pitch * roll) → quaternion [w, x, y, z]
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    z = torch.rand(num, device=dev) * (z_max - z_min) + z_min
    z_stand = torch.rand(num, device=dev) * (standing_z_max - standing_z_min) + standing_z_min
    z = torch.where(is_stand, z_stand, z)

    env.sim.data.qpos[env_ids, 2] = z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    asset: Entity = env.scene[asset_cfg.name]
    servo_ids = _servo_joint_ids(env, asset)
    cols = torch.tensor([7 + j for j in servo_ids], device=dev, dtype=torch.long)
    pose = torch.tensor(joint_pose, device=dev, dtype=torch.float32).unsqueeze(0).expand(num, -1).clone()
    home = asset.data.default_joint_pos[env_ids.long()][:, servo_ids]
    pose = torch.where(is_stand.unsqueeze(1), home, pose)
    if joint_noise_std > 0.0:
        pose = pose + torch.randn_like(pose) * joint_noise_std
    env.sim.data.qpos[env_ids.unsqueeze(1).long(), cols.unsqueeze(0)] = pose
    # Zero joint velocities too (qvel columns = 6 + entity joint index).
    vcols = torch.tensor([6 + j for j in servo_ids], device=dev, dtype=torch.long)
    env.sim.data.qvel[env_ids.unsqueeze(1).long(), vcols.unsqueeze(0)] = 0.0


def _foot_found(env: ManagerBasedRlEnv, sensor_name: str, slot: int) -> torch.Tensor:
    """Binary contact flag of one foot from the two-foot ``feet_ground_contact``
    sensor (slot 0 = left, 1 = right — the pattern order in the sensor cfg)."""
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() > 1:
        found = found[:, slot]
    return torch.clamp(torch.nan_to_num(found.float(), nan=0.0), 0.0, 1.0)


def foot_contact_reward(env: ManagerBasedRlEnv, sensor_name: str, slot: int) -> torch.Tensor:
    """+1 while the given foot touches the ground (pin the stance foot, anti-hop)."""
    return _foot_found(env, sensor_name, slot)


def foot_contact_penalty(env: ManagerBasedRlEnv, sensor_name: str, slot: int) -> torch.Tensor:
    """−1 while the given foot touches the ground (self-negating → POSITIVE weight).
    The swing foot touching down is the SOFT failure: cheap, never terminal."""
    return -_foot_found(env, sensor_name, slot)


def _robot_com_xy(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    """Horizontal WHOLE-ROBOT centre of mass (MuJoCo ``subtree_com`` of the root
    body). ``asset.data.root_com_pos_w`` is the trunk body's own CoM (xipos),
    which with a 38 %-mass head and a lifted leg sits centimetres away from the
    true CoM the poses were validated with (notes/tools/duck_pose.py)."""
    com = env.sim.data.subtree_com[:, asset.indexing.root_body_id, :2]
    return torch.nan_to_num(com, nan=0.0)


def com_over_foot(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    std: float = 0.02,
) -> torch.Tensor:
    """Ungated Gaussian on the horizontal distance CoM ↔ stance-foot site
    (``asset_cfg.site_names=["right_foot"]``). Same quantity as
    ``com_over_support_foot`` minus the kick phase gate."""
    asset: Entity = env.scene[asset_cfg.name]
    com_xy = _robot_com_xy(env, asset)
    foot_xy = asset.data.site_pos_w[:, asset_cfg.site_ids[0], :2]
    dist2 = torch.nan_to_num(((com_xy - foot_xy) ** 2).sum(dim=-1), nan=1.0)
    return torch.exp(-dist2 / (std ** 2))


def foot_height_gaussian(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    target: float = 0.05,
    std: float = 0.03,
    gate_sensor_name: str | None = None,
    gate_slot: int = 1,
) -> torch.Tensor:
    """Gaussian on the swing-foot site height above the terrain around ``target``
    (m); gated on the stance foot being in contact so hopping cannot farm it."""
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2] - env.scene.terrain.env_origins[:, 2]
    z = torch.nan_to_num(z, nan=0.0)
    reward = torch.exp(-((z - target) / std) ** 2)
    if gate_sensor_name is not None:
        reward = reward * _foot_found(env, gate_sensor_name, gate_slot)
    return reward


def projected_gravity_match(
    env: ManagerBasedRlEnv,
    target: tuple = (0.0, 0.0, -1.0),
    std: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian on ‖g_b − target‖: rewards holding a specific trunk lean (the
    flamingo trunk is tilted ~24°, so a plain upright reward would fight the pose)."""
    asset: Entity = env.scene[asset_cfg.name]
    g = torch.nan_to_num(asset.data.projected_gravity_b, nan=0.0)
    t = torch.tensor(target, device=env.device, dtype=g.dtype).unsqueeze(0)
    err2 = ((g - t) ** 2).sum(dim=-1)
    return torch.exp(-err2 / (std ** 2))


def lateral_tilt_penalty(
    env: ManagerBasedRlEnv,
    threshold: float = 0.45,
    direction: float = -1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Quadratic penalty (≤ 0, self-negating → POSITIVE weight) when the trunk
    rolls further toward ``direction`` (sign of g_b,y) than ``threshold``.
    Falling toward the stance side has no catch — the HARD failure direction —
    so it is taxed early and steeply, unlike the swing side."""
    asset: Entity = env.scene[asset_cfg.name]
    gy = torch.nan_to_num(asset.data.projected_gravity_b[:, 1], nan=0.0)
    over = torch.clamp(direction * gy - threshold, min=0.0)
    return -(over ** 2)


def joint_vel_gaussian(
    env: ManagerBasedRlEnv,
    std: float = 1.0,
    gate_sensor_name: str | None = None,
    gate_slot: int = 1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Stillness: exp(−mean(q̇²)/std²) over the servo joints, optionally gated on
    the stance-foot contact (quiet balancing pays; lying still on the floor does not
    because the fall terminates / the stance foot is gone)."""
    asset: Entity = env.scene[asset_cfg.name]
    qd = torch.nan_to_num(_servo_joint_vel(env, asset), nan=0.0)
    reward = torch.exp(-(qd ** 2).mean(dim=-1) / (std ** 2))
    if gate_sensor_name is not None:
        reward = reward * _foot_found(env, gate_sensor_name, gate_slot)
    return reward


def reward_param_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    param_stages: list[dict],
) -> torch.Tensor:
    """Mutate a reward term's params at scheduled steps (twin of
    ``event_param_curriculum``; uses the live RewardManager term cfg)."""
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    current = param_stages[0]["params"]
    for stage in param_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["params"]
    term_cfg.params.update(current)
    first_val = next(iter(current.values()))
    return torch.tensor(float(first_val) if isinstance(first_val, (int, float)) else 0.0)


class FlamingoCommand(UniformVelocityCommand):
    """cmd = [flag, side, 0] with dwell-time resampling and a slewed target blend.

    Resampling: flag ~ Bernoulli(cfg.flamingo_prob). The side is re-drawn only
    when the previous command was STAND (or at episode start) — flipping the
    stance foot mid-hold is not a transition we train. ``alpha`` slews toward
    the flag at 1/ramp_s per second; the spawn event initialises it from the
    spawn bucket (1 in-pose, 0 standing) via ``set_alpha``.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._flamingo_prob = float(getattr(cfg, "flamingo_prob", 0.6))
        self._ramp_s = float(getattr(cfg, "ramp_s", 1.5))
        self._env_ref = env
        self._alpha = torch.zeros(self.num_envs, device=self.device)
        self._fresh = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        # Side requested by the spawn event (0 = none). mjlab applies reset EVENTS
        # before command_manager.reset(), so the event that teleports the robot
        # into a one-foot pose decides the side and the resample honours it.
        self._pending_side = torch.zeros(self.num_envs, device=self.device)
        # Latched stance side (±1) the REWARDS use; never 0. The OBSERVED side
        # slot is zeroed (p = zero_side_prob) while flag = 0 so the runtime's
        # all-zero idle command [0, 0, 0] is in-distribution.
        self._stance_side = torch.ones(self.num_envs, device=self.device)
        self._zero_side_prob = float(getattr(cfg, "zero_side_prob", 0.5))

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    @property
    def alpha(self) -> torch.Tensor:
        return self._alpha

    @property
    def flag(self) -> torch.Tensor:
        return self.vel_command_b[:, 0]

    @property
    def side(self) -> torch.Tensor:
        """Observed side slot (0 allowed while standing)."""
        return self.vel_command_b[:, 1]

    @property
    def stance_side(self) -> torch.Tensor:
        """Latched ±1 stance side for the rewards (last non-zero side)."""
        return self._stance_side

    def set_alpha(self, env_ids: torch.Tensor, value: torch.Tensor) -> None:
        self._alpha[env_ids] = value

    def request_side(self, env_ids: torch.Tensor, side: torch.Tensor) -> None:
        """Called by the spawn event: pin the side for the upcoming resample and
        expose it immediately in the command buffer."""
        self._pending_side[env_ids] = side
        self._stance_side[env_ids] = side
        self.vel_command_b[env_ids, 1] = side

    def reset(self, env_ids=None):
        if env_ids is not None and len(env_ids) > 0:
            self._fresh[env_ids] = True
        return super().reset(env_ids)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        prev_flag = self.vel_command_b[env_ids, 0]
        prev_side = self._stance_side[env_ids]
        new_flag = (torch.rand(n, device=self.device) < self._flamingo_prob).float()
        new_side = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        # Re-draw the side only from STAND (or at episode start); a hold keeps it.
        redraw = self._fresh[env_ids] | (prev_flag < 0.5)
        side = torch.where(redraw, new_side, prev_side)
        pending = self._pending_side[env_ids]
        side = torch.where(pending.abs() > 0.5, pending, side)
        self._pending_side[env_ids] = 0.0
        self._stance_side[env_ids] = side
        # Observed side: the latched side on one foot; while standing, zero it
        # with p = zero_side_prob (deployment idle parity) else keep it.
        zero = (new_flag < 0.5) & (torch.rand(n, device=self.device) < self._zero_side_prob)
        obs_side = torch.where(zero, torch.zeros_like(side), side)
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = new_flag
        self.vel_command_b[env_ids, 1] = obs_side
        self._fresh[env_ids] = False

    def compute(self, dt: float) -> None:
        super().compute(dt)
        step = dt / max(self._ramp_s, 1e-6)
        delta = self.vel_command_b[:, 0] - self._alpha
        self._alpha += torch.clamp(delta, -step, step)

    def _update_command(self) -> None:
        pass

    def _update_metrics(self) -> None:
        pass


@_dataclass(kw_only=True)
class FlamingoCommandCfg(UniformVelocityCommandCfg):
    class_type: type = FlamingoCommand
    flamingo_prob: float = 0.6
    ramp_s: float = 1.5
    # P(observed side slot = 0 | flag = 0): trains the all-zero idle command.
    zero_side_prob: float = 0.5

    def build(self, env: ManagerBasedRlEnv) -> "FlamingoCommand":
        return FlamingoCommand(self, env)


def _smoothstep(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    t = torch.clamp((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _fl_ctx(env: ManagerBasedRlEnv, command_name: str):
    """(flag, latched ±1 stance side, alpha) of the FlamingoCommand term
    (falls back to the raw command slots)."""
    term = env.command_manager.get_term(command_name)
    cmd = env.command_manager.get_command(command_name)
    flag = cmd[:, 0]
    side = getattr(term, "stance_side", None)
    if side is None:
        side = torch.where(cmd[:, 1] < 0.0, -1.0, 1.0)
    alpha = getattr(term, "alpha", None)
    if alpha is None:
        alpha = flag
    return flag, side, alpha


def _fl_feet_found(env: ManagerBasedRlEnv, sensor_name: str, side: torch.Tensor):
    """(stance_found, swing_found) for the per-env side (+1 right stance)."""
    left = _foot_found(env, sensor_name, 0)
    right = _foot_found(env, sensor_name, 1)
    right_stance = side > 0
    stance = torch.where(right_stance, right, left)
    swing = torch.where(right_stance, left, right)
    return stance, swing


def _fl_site_xyz(env: ManagerBasedRlEnv, left_cfg: SceneEntityCfg, right_cfg: SceneEntityCfg):
    asset: Entity = env.scene[left_cfg.name]
    left = torch.nan_to_num(asset.data.site_pos_w[:, left_cfg.site_ids[0], :], nan=0.0)
    right = torch.nan_to_num(asset.data.site_pos_w[:, right_cfg.site_ids[0], :], nan=0.0)
    return asset, left, right


def set_flamingo_cycle_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    pose_right: list,
    pose_left: list,
    base_roll: float,
    base_pitch: float,
    z_min: float,
    z_max: float,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    tilt_noise: float = 0.0,
    joint_noise_std: float = 0.0,
    in_pose_prob: float = 0.5,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
):
    """Spawn bucket for the cycle env: with ``in_pose_prob`` IN the one-foot pose
    of a side drawn here and pinned into the command term (trunk rolled
    ±base_roll), else standing at HOME. Sets the command's alpha (1 in-pose / 0 standing) so the
    shaping targets start where the robot is. The flag is left as the command
    term sampled it → all four (spawn × command) combinations are trained,
    including "lower the foot from a static hold" and "hold what you do"."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)
    dev = env.device
    term = env.command_manager.get_term(command_name)
    # The event decides the side (reset events run BEFORE the command reset in
    # mjlab's _reset_idx) and pins it in the command term.
    side = torch.where(torch.rand(num, device=dev) < 0.5, -1.0, 1.0)
    if hasattr(term, "request_side"):
        term.request_side(env_ids.long(), side)

    in_pose = torch.rand(num, device=dev) < in_pose_prob
    yaw = torch.rand(num, device=dev) * 2 * np.pi - np.pi
    roll = side * float(base_roll)
    pitch = torch.full((num,), float(base_pitch), device=dev)
    if tilt_noise > 0.0:
        roll = roll + (torch.rand(num, device=dev) * 2 - 1) * tilt_noise
        pitch = pitch + (torch.rand(num, device=dev) * 2 - 1) * tilt_noise
    roll = torch.where(in_pose, roll, torch.zeros_like(roll))
    pitch = torch.where(in_pose, pitch, torch.zeros_like(pitch))
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    z = torch.rand(num, device=dev) * (z_max - z_min) + z_min
    z_stand = torch.rand(num, device=dev) * (standing_z_max - standing_z_min) + standing_z_min
    z = torch.where(in_pose, z, z_stand)

    env.sim.data.qpos[env_ids, 2] = z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    asset: Entity = env.scene[asset_cfg.name]
    servo_ids = _servo_joint_ids(env, asset)
    cols = torch.tensor([7 + j for j in servo_ids], device=dev, dtype=torch.long)
    pr = torch.tensor(pose_right, device=dev, dtype=torch.float32).unsqueeze(0).expand(num, -1)
    pl = torch.tensor(pose_left, device=dev, dtype=torch.float32).unsqueeze(0).expand(num, -1)
    pose = torch.where((side > 0).unsqueeze(1), pr, pl).clone()
    home = asset.data.default_joint_pos[env_ids.long()][:, servo_ids]
    pose = torch.where(in_pose.unsqueeze(1), pose, home)
    if joint_noise_std > 0.0:
        pose = pose + torch.randn_like(pose) * joint_noise_std
    env.sim.data.qpos[env_ids.unsqueeze(1).long(), cols.unsqueeze(0)] = pose
    vcols = torch.tensor([6 + j for j in servo_ids], device=dev, dtype=torch.long)
    env.sim.data.qvel[env_ids.unsqueeze(1).long(), vcols.unsqueeze(0)] = 0.0

    if hasattr(term, "set_alpha"):
        term.set_alpha(env_ids.long(), in_pose.float())


def fl_com_target(
    env: ManagerBasedRlEnv,
    command_name: str,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
    std: float = 0.02,
) -> torch.Tensor:
    """Gaussian on CoM_xy vs the BLENDED support target: (1−α)·midpoint(feet) +
    α·stance foot. The leader of the stack: the weight shift IS the transition."""
    flag, side, alpha = _fl_ctx(env, command_name)
    asset, left, right = _fl_site_xyz(env, left_cfg, right_cfg)
    stance = torch.where((side > 0).unsqueeze(1), right, left)
    mid = 0.5 * (left + right)
    a = alpha.unsqueeze(1)
    target_xy = (1.0 - a) * mid[:, :2] + a * stance[:, :2]
    com_xy = _robot_com_xy(env, asset)
    dist2 = ((com_xy - target_xy) ** 2).sum(dim=-1)
    return torch.exp(-dist2 / (std ** 2))


def fl_stance_foot_grounded(env: ManagerBasedRlEnv, command_name: str, sensor_name: str) -> torch.Tensor:
    """+1 while the (side's) stance foot touches the floor — in both modes."""
    _, side, _ = _fl_ctx(env, command_name)
    stance, _ = _fl_feet_found(env, sensor_name, side)
    return stance


def fl_swing_foot_contact(
    env: ManagerBasedRlEnv, command_name: str, sensor_name: str,
    lo: float = 0.4, hi: float = 0.9,
) -> torch.Tensor:
    """Swing-foot contact: +1 when standing is commanded (α ≤ lo), −1 once the
    lift is due (α ≥ hi), neutral in between. Mixed sign by design."""
    _, side, alpha = _fl_ctx(env, command_name)
    _, swing = _fl_feet_found(env, sensor_name, side)
    return swing * (1.0 - 2.0 * _smoothstep(alpha, lo, hi))


def fl_swing_foot_clear(
    env: ManagerBasedRlEnv,
    command_name: str,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
    sensor_name: str,
    target: float = 0.05,
    std: float = 0.03,
    lo: float = 0.6,
    hi: float = 1.0,
) -> torch.Tensor:
    """Gaussian on the swing-foot height around ``target``, gated on stance
    contact (no hopping) and on the lift being due (smoothstep of α)."""
    _, side, alpha = _fl_ctx(env, command_name)
    _, left, right = _fl_site_xyz(env, left_cfg, right_cfg)
    swing = torch.where((side > 0).unsqueeze(1), left, right)
    z = swing[:, 2] - env.scene.terrain.env_origins[:, 2]
    reward = torch.exp(-((z - target) / std) ** 2)
    stance_found, _ = _fl_feet_found(env, sensor_name, side)
    return reward * stance_found * _smoothstep(alpha, lo, hi)


def fl_pose_track(
    env: ManagerBasedRlEnv,
    command_name: str,
    pose_right: list,
    pose_left: list,
    std: float = 0.5,
    mid_attenuation: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian pose match vs the blended target (1−α)·HOME + α·FLAMINGO(side).
    Attenuated mid-transition (×(1 − mid_attenuation) at α = 0.5): the linear
    joint blend is a hint, the CoM term leads the actual weight shift."""
    _, side, alpha = _fl_ctx(env, command_name)
    asset = env.scene[asset_cfg.name]
    home = _servo_default_joint_pos(env, asset)
    pr = torch.tensor(pose_right, device=env.device, dtype=home.dtype).unsqueeze(0)
    pl = torch.tensor(pose_left, device=env.device, dtype=home.dtype).unsqueeze(0)
    fl_pose = torch.where((side > 0).unsqueeze(1), pr, pl)
    a = alpha.unsqueeze(1)
    target = (1.0 - a) * home + a * fl_pose
    q = torch.nan_to_num(_servo_joint_pos(env, asset), nan=0.0)
    match = torch.exp(-((q - target) / std) ** 2).mean(dim=-1)
    factor = 1.0 - mid_attenuation * 4.0 * alpha * (1.0 - alpha)
    return match * factor


def fl_gravity_track(
    env: ManagerBasedRlEnv,
    command_name: str,
    gravity_right: tuple,
    gravity_left: tuple,
    std: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian on ‖g_b − target‖ with target = normalize((1−α)·(0,0,−1) + α·g(side)):
    upright when standing, the pose's lean on one foot, leaning in between."""
    _, side, alpha = _fl_ctx(env, command_name)
    asset = env.scene[asset_cfg.name]
    g = torch.nan_to_num(asset.data.projected_gravity_b, nan=0.0)
    gr = torch.tensor(gravity_right, device=env.device, dtype=g.dtype).unsqueeze(0)
    gl = torch.tensor(gravity_left, device=env.device, dtype=g.dtype).unsqueeze(0)
    gf = torch.where((side > 0).unsqueeze(1), gr, gl)
    up = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=g.dtype).unsqueeze(0)
    a = alpha.unsqueeze(1)
    t = (1.0 - a) * up + a * gf
    t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    err2 = ((g - t) ** 2).sum(dim=-1)
    return torch.exp(-err2 / (std ** 2))


def fl_stillness(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    std: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Joint-velocity Gaussian, paid only once the ramp is complete (|flag − α|
    small) and the stance foot is down: quiet rests in BOTH modes, nothing
    mid-transition."""
    flag, side, alpha = _fl_ctx(env, command_name)
    asset = env.scene[asset_cfg.name]
    qd = torch.nan_to_num(_servo_joint_vel(env, asset), nan=0.0)
    reward = torch.exp(-(qd ** 2).mean(dim=-1) / (std ** 2))
    ramp_done = ((flag - alpha).abs() < 0.02).float()
    stance_found, _ = _fl_feet_found(env, sensor_name, side)
    return reward * ramp_done * stance_found


def fl_stance_side_tilt(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float = 0.45,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """≤ 0 (self-negating → POSITIVE weight): quadratic tax on rolling further
    toward the stance side than ``threshold`` (|g_b,y|). Right stance: g_b,y more
    negative than −thr; left stance: more positive than +thr. Falling that way
    has no catch."""
    _, side, _ = _fl_ctx(env, command_name)
    asset = env.scene[asset_cfg.name]
    gy = torch.nan_to_num(asset.data.projected_gravity_b[:, 1], nan=0.0)
    over = torch.clamp(-side * gy - threshold, min=0.0)
    return -(over ** 2)


def fl_single_support_success(
    env: ManagerBasedRlEnv, command_name: str, sensor_name: str,
) -> torch.Tensor:
    """Diagnostic (log-only, weight 0 is fine): 1 when the commanded state is
    realised — one foot down and the other up in flamingo, both down in stand."""
    flag, side, _ = _fl_ctx(env, command_name)
    stance, swing = _fl_feet_found(env, sensor_name, side)
    one_foot = stance * (1.0 - swing)
    two_feet = stance * swing
    return torch.where(flag > 0.5, one_foot, two_feet)
