from math import pi
import torch
import torch.nn.functional as F
import torch.distributions as D
import math
import warp as wp
from typing import Sequence, Tuple
from typing_extensions import override

from active_adaptation.utils.math import (
    quat_from_euler_xyz,
    quat_rotate, 
    quat_rotate_inverse,
    clamp_norm,
    yaw_quat,
    yaw_rotate,
    wrap_to_pi,
    MultiUniform,
    sample_quat_yaw
)
import active_adaptation.utils.symmetry as symmetry_utils
from active_adaptation.envs.mdp.base import Command

def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low


def quat_to_yaw(quat: torch.Tensor):
    q_w, q_x, q_y, q_z = quat.unbind(-1)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)
    return yaw % (2 * torch.pi)

class KpPosTrack(Command):

    def __init__(
        self,
        env,
        linvel_x_range=(-1.0, 1.0),
        linvel_y_range=(-1.0, 1.0),
        angvel_range=(-1, 1),
        yaw_stiffness_range=(0.5, 0.6),
        use_stiffness_ratio: float = 0.5,
        base_height_range=(0.2, 0.4),
        resample_interval: int = 300,
        resample_prob: float = 0.75,
        stand_prob=0.2,
        target_yaw_range=(0, torch.pi * 2),
        curriculum: bool = False,
        teleop: bool = False,
        # Hand end-effector target parameters
        hand_body_names: Tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
        hand_target_range_b: Tuple[Tuple[float, ...], Tuple[float, ...]] = (
            (-0.1, -0.3, -0.2),  # min (x, y, z) relative to torso in body frame
            (0.5, 0.3, 0.5),     # max (x, y, z) relative to torso in body frame
        ),
        hand_resample_interval: int = 200,
        hand_max_step_size: float = 0.005,  # max distance per step (m), controls trajectory speed
    ):
        super().__init__(env, teleop)
        self.linvel_x_range = linvel_x_range
        self.linvel_y_range = linvel_y_range
        self.angvel_range = angvel_range
        self.use_stiffness_ratio = use_stiffness_ratio
        self.yaw_stiffness_range = yaw_stiffness_range
        self.base_height_range = base_height_range
        self.resample_interval = resample_interval
        self.resample_prob = resample_prob
        self.stand_prob = stand_prob
        self.curriculum = curriculum and self.env.backend == "isaac"

        # Hand target setup
        self.hand_body_ids, self.hand_body_names_resolved = self.asset.find_bodies(
            "|".join(hand_body_names)
        )
        assert len(self.hand_body_ids) == 2, (
            f"Expected 2 hand bodies, found {len(self.hand_body_ids)}: {self.hand_body_names_resolved}"
        )
        # Also find a torso body as reference for relative sampling
        self.torso_body_id = self.asset.find_bodies("torso_link")[0][0]
        self.hand_target_min_b = torch.tensor(hand_target_range_b[0], device=self.device)
        self.hand_target_max_b = torch.tensor(hand_target_range_b[1], device=self.device)
        self.hand_resample_interval = hand_resample_interval
        self.hand_max_step_size = hand_max_step_size

        if self.curriculum:
            self.terrain = self.env.scene.terrain
            assert self.terrain.cfg.terrain_type == "generator", "Curriculum is only supported for generator terrain"
            assert self.terrain.cfg.terrain_generator.curriculum, "Curriculum is not enabled for the terrain"

        with torch.device(self.device):
            if all(isinstance(r, Sequence) for r in target_yaw_range):
                self.target_yaw_dist = MultiUniform(torch.tensor(target_yaw_range))
            else:
                self.target_yaw_dist = D.Uniform(*torch.tensor(target_yaw_range))

            self.target_yaw = torch.zeros(self.num_envs, 1)
            self.yaw_stiffness = torch.zeros(self.num_envs, 1)
            self.use_stiffness = torch.zeros(self.num_envs, 1, dtype=bool)
            self.fixed_yaw_speed = torch.zeros(self.num_envs, 1)

            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)

            self.command_speed = torch.zeros(self.num_envs, 1)
            self.next_command_linvel = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_b = torch.zeros(self.num_envs, 3)
            self.cmd_linvel_w = torch.zeros(self.num_envs, 3)
            self.cmd_yawvel_b = torch.zeros(self.num_envs, 1)
            self.cmd_base_height = torch.zeros(self.num_envs, 1)

            self.distance_commanded = torch.zeros(self.num_envs, 1)
            self.distance_traveled = torch.zeros(self.num_envs, 1)

            self.cum_error = torch.zeros(self.num_envs, 2)
            self._cum_linvel_error = self.cum_error[:, 0].unsqueeze(1)
            self._cum_angvel_error = self.cum_error[:, 1].unsqueeze(1)

            # Hand end-effector trajectory buffers (body frame, relative to torso)
            # Trajectory: linearly interpolate from start to end, duration varies per env
            self.hand_traj_start_b = torch.zeros(self.num_envs, 2, 3)  # trajectory start
            self.hand_traj_end_b = torch.zeros(self.num_envs, 2, 3)    # trajectory end
            self.hand_traj_step = torch.zeros(self.num_envs, 1, dtype=torch.long)  # current step
            self.hand_traj_duration = torch.ones(self.num_envs, 1, dtype=torch.long)  # per-env duration
            # Interpolated target at current progress
            self.target_hand_pos_b = torch.zeros(self.num_envs, 2, 3)
            # Computed errors in body frame
            self.hand_pos_error_b = torch.zeros(self.num_envs, 2, 3)

        if self.teleop and self.env.backend == "isaac":
            self.key_mappings_pos = {
                "W": torch.tensor(
                    [self.linvel_x_range[1], 0.0, 0.0], device=self.device
                ),
                "S": torch.tensor(
                    [self.linvel_x_range[0], 0.0, 0.0], device=self.device
                ),
                "A": torch.tensor(
                    [0.0, self.linvel_y_range[1], 0.0], device=self.device
                ),
                "D": torch.tensor(
                    [0.0, self.linvel_y_range[0], 0.0], device=self.device
                ),
            }
            # use left-right arrow keys to rotate
            self.key_mappings_yaw = {
                "LEFT": torch.tensor([self.angvel_range[1]], device=self.device),
                "RIGHT": torch.tensor([self.angvel_range[0]], device=self.device),
            }
            # state for teleoperation commands (shared across all envs)
            self._teleop_linvel = torch.zeros(3, device=self.device)
            self._teleop_yaw = torch.zeros(1, device=self.device)
            # speed modifiers controlled by shift/ctrl
            self._speed_scale = 0.8
            self._fast_speed_scale = 1.6
            self._slow_speed_scale = 0.4
            from active_adaptation.utils.isaac_keyboard import IsaacKeyboardManager
            self.keyboard_manager = IsaacKeyboardManager()
        
        if self.env.sim.has_gui():
            if self.env.backend == "mjlab":
                from active_adaptation.viewer import MjLabViewer
                self.viewer: MjLabViewer = self.env.sim.viewer
                self.axes_handle = self.viewer.add_batched_axes("target_yaw")
                self.lines_handle = self.viewer.add_line_segments("cmd_linvel_w", (1., 0., 0.))
                self.lines_handle.line_width = 2.0
    
    @property
    def command(self):
        return torch.cat([
            self.cmd_linvel_b[:, :2],
            self.cmd_yawvel_b.reshape(self.num_envs, 1),
            self.cmd_base_height.reshape(self.num_envs, 1),
        ], dim=-1)

    @override
    def sample_init(self, env_ids):
        if self.curriculum and self.env.episode_count > 1: # and self.env.training:
            distance_traveled = self.distance_traveled[env_ids]
            distance_commanded = self.distance_commanded[env_ids].clamp_min(1.0)
            move_up = distance_traveled > distance_commanded * 0.8
            move_down = distance_traveled < distance_commanded * 0.4
            move_up = move_up & ~move_down
            self.terrain.update_env_origins(env_ids, move_up.squeeze(-1), move_down.squeeze(-1))
            self._origins = self.terrain.env_origins.clone()
            self.env.extra["curriculum/terrain_level"] = self.terrain.terrain_levels.float().mean()
        self.env.extra["curriculum/distance_commanded"] = self.distance_commanded.mean()
        self.env.extra["curriculum/distance_traveled"] = self.distance_traveled.mean()
        self.distance_commanded[env_ids] = 0.0
        self.distance_traveled[env_ids] = 0.0
        return super().sample_init(env_ids)

    @override
    def reset(self, env_ids):
        self.next_command_linvel[env_ids] = 0.0
        self.cmd_linvel_b[env_ids] = 0.0
        self.target_yaw[env_ids] = self.asset.data.heading_w[env_ids, None]
        self.cmd_yawvel_b[env_ids] = 0.0

        self._cum_linvel_error[env_ids] = 0.0
        self._cum_angvel_error[env_ids] = 0.0
        self.is_standing_env[env_ids] = True

        # Initialize hand targets to current hand positions (error = 0)
        self._init_hand_targets(env_ids)

    @override
    def update(self):
        if self.teleop:
            self._update_teleop()
        else:
            self._update_training()
    
    def _update_teleop(self):
        if self.env.backend != "isaac":
            # fall back to training behaviour when not using Isaac backend
            return self._update_training()

        km = self.keyboard_manager.key_pressed
        if (km.get("LEFT_SHIFT") or km.get("RIGHT_SHIFT")):
            scale = self._fast_speed_scale
        elif (km.get("LEFT_CONTROL") or km.get("RIGHT_CONTROL")):
            scale = self._slow_speed_scale
        else:
            scale = self._speed_scale

        self._teleop_linvel.zero_()
        for key, vel in self.key_mappings_pos.items():
            if km.get(key, False):
                self._teleop_linvel.add_(vel)
        self._teleop_yaw.zero_()
        for key, vel in self.key_mappings_yaw.items():
            if km.get(key, False):
                self._teleop_yaw.add_(vel)

        linvel = (self._teleop_linvel * scale).unsqueeze(0).expand(self.num_envs, -1)
        linvel[:, 2] = 0.0
        max_speed = max(0.0, 2.5 - self._teleop_yaw.abs().item())
        self.cmd_linvel_b = clamp_norm(linvel, max=max_speed)
        self.cmd_yawvel_b[:] = (self._teleop_yaw * scale).clamp(*self.angvel_range)
        self.cmd_base_height[:] = sum(self.base_height_range) / 2

        self.quat_w = self.asset.data.root_link_quat_w
        self.cmd_linvel_w = quat_rotate(yaw_quat(self.quat_w), self.cmd_linvel_b)
        self.command_speed = self.cmd_linvel_b.norm(dim=-1, keepdim=True)
        self.is_standing_env = (self.command_speed < 0.1) & (self.cmd_yawvel_b.abs() < 0.1)

    def _update_training(self):
        self.body_heading_w = self.asset.data.heading_w.unsqueeze(1)
        self.lin_vel_w = self.asset.data.root_com_lin_vel_w
        self.ang_vel_w = self.asset.data.root_com_ang_vel_w
        self.quat_w = self.asset.data.root_link_quat_w

        # this is used for terminating episodes where the robot is inactive due to whatever reason
        linvel_diff = self.lin_vel_w[:, :2] - self.cmd_linvel_w[:, :2]
        linvel_error = linvel_diff.norm(dim=-1, keepdim=True)
        angvel_diff = self.cmd_yawvel_b - self.ang_vel_w[:, 2:3]
        angvel_error = angvel_diff.abs()

        self._cum_linvel_error.mul_(0.98).add_(linvel_error * self.env.step_dt)
        self._cum_angvel_error.mul_(0.98).add_(angvel_error * self.env.step_dt)

        max_command_speed = (2.5 - self.cmd_yawvel_b.abs()).clamp(0.0)
        self.cmd_linvel_b.lerp_(self.next_command_linvel, 0.1)
        self.cmd_linvel_b = clamp_norm(self.cmd_linvel_b, max=max_command_speed)
        self.command_speed = self.cmd_linvel_b.norm(dim=-1, keepdim=True)
    
        self.current_speed = self.lin_vel_w.norm(dim=-1, keepdim=True)
        self.distance_commanded = self.distance_commanded + self.command_speed * self.env.step_dt
        self.distance_traveled = self.distance_traveled + self.current_speed * self.env.step_dt

        interval_reached = (self.env.episode_length_buf - 20) % self.resample_interval == 0
        resample_vel = interval_reached & (
            self.with_prob(self.num_envs, self.resample_prob)
            | self.is_standing_env.squeeze(1)
        )
        resample_yaw = interval_reached & (
            self.with_prob(self.num_envs, self.resample_prob)
            | self.is_standing_env.squeeze(1)
        )
        self.sample_vel_command(resample_vel.nonzero().squeeze(-1))
        self.sample_yaw_command(resample_yaw.nonzero().squeeze(-1))

        yaw_diff = wrap_to_pi(self.target_yaw - self.body_heading_w).reshape(self.num_envs, 1)
        cmd_yawvel_b = torch.clamp(
            self.yaw_stiffness * yaw_diff,
            min=self.angvel_range[0],
            max=self.angvel_range[1],
        ).reshape(self.num_envs, 1)

        self.cmd_yawvel_b = torch.where(
            self.use_stiffness,
            cmd_yawvel_b,
            self.fixed_yaw_speed
        ).reshape(self.num_envs, 1)

        self.cmd_linvel_w = quat_rotate(yaw_quat(self.quat_w), self.cmd_linvel_b)
        self.is_standing_env = (self.command_speed < 0.1) & (self.cmd_yawvel_b.abs() < 0.1)

        # Update hand target tracking
        self._update_hand_targets()

    def _update_hand_targets(self):
        """Advance trajectory interpolation, compute errors, and resample when segment ends."""
        # Advance trajectory step
        self.hand_traj_step += 1

        # Compute interpolation progress: [0, 1], per-env duration
        progress = (self.hand_traj_step.float() / self.hand_traj_duration.float()).clamp(0.0, 1.0)  # (N, 1)
        # Interpolate target: start + progress * (end - start)
        self.target_hand_pos_b = (
            self.hand_traj_start_b + progress.unsqueeze(-1) * (self.hand_traj_end_b - self.hand_traj_start_b)
        )

        # Get current hand positions in world frame: (N, 2, 3)
        hand_pos_w = self.asset.data.body_link_pos_w[:, self.hand_body_ids]
        # Get torso position as reference
        torso_pos_w = self.asset.data.body_link_pos_w[:, self.torso_body_id]  # (N, 3)
        # Hand positions relative to torso in world frame
        hand_rel_w = hand_pos_w - torso_pos_w.unsqueeze(1)  # (N, 2, 3)
        # Rotate to body frame
        quat_w = self.asset.data.root_link_quat_w
        yaw_q = yaw_quat(quat_w)  # (N, 4)
        hand_rel_b_left = quat_rotate_inverse(yaw_q, hand_rel_w[:, 0])   # (N, 3)
        hand_rel_b_right = quat_rotate_inverse(yaw_q, hand_rel_w[:, 1])  # (N, 3)
        current_hand_b = torch.stack([hand_rel_b_left, hand_rel_b_right], dim=1)  # (N, 2, 3)
        # Compute error: target - current (body frame)
        self.hand_pos_error_b = self.target_hand_pos_b - current_hand_b

        # Resample new trajectory segment when current one ends
        segment_done = (self.hand_traj_step >= self.hand_traj_duration).squeeze(-1)
        resample_ids = segment_done.nonzero().squeeze(-1)
        if len(resample_ids) > 0:
            self.sample_hand_command(resample_ids)

    def sample_vel_command(self, env_ids: torch.Tensor):
        next_command_linvel = torch.zeros(len(env_ids), 3, device=self.device)
        next_command_linvel[:, 0].uniform_(*self.linvel_x_range)
        next_command_linvel[:, 1].uniform_(*self.linvel_y_range)

        speed = next_command_linvel.norm(dim=-1, keepdim=True)
        r = torch.rand(len(env_ids), 1, device=self.device) < self.stand_prob
        valid = ~((speed < 0.10) | r)
        self.next_command_linvel[env_ids] = next_command_linvel * valid
        self.cmd_base_height[env_ids] = sample_uniform((len(env_ids), 1), *self.base_height_range, self.device)

    def sample_yaw_command(self, env_ids: torch.Tensor):
        self.target_yaw[env_ids] = self.target_yaw_dist.sample(env_ids.shape).unsqueeze(1)
        shape = (len(env_ids), 1)
        self.yaw_stiffness[env_ids] = sample_uniform(shape, *self.yaw_stiffness_range, self.device)
        self.use_stiffness[env_ids] = self.with_prob(shape, self.use_stiffness_ratio)
        self.fixed_yaw_speed[env_ids] = sample_uniform(shape, *self.angvel_range, self.device)

    def _init_hand_targets(self, env_ids: torch.Tensor):
        """Set hand trajectory start=end=current positions so error starts at 0."""
        hand_pos_w = self.asset.data.body_link_pos_w[env_ids][:, self.hand_body_ids]
        torso_pos_w = self.asset.data.body_link_pos_w[env_ids][:, self.torso_body_id]
        hand_rel_w = hand_pos_w - torso_pos_w.unsqueeze(1)
        quat_w = self.asset.data.root_link_quat_w[env_ids]
        yaw_q = yaw_quat(quat_w)
        hand_rel_b_left = quat_rotate_inverse(yaw_q, hand_rel_w[:, 0])
        hand_rel_b_right = quat_rotate_inverse(yaw_q, hand_rel_w[:, 1])
        current_b = torch.stack([hand_rel_b_left, hand_rel_b_right], dim=1)  # (n, 2, 3)
        # Initialize trajectory: start = end = current → stationary
        self.hand_traj_start_b[env_ids] = current_b
        self.hand_traj_end_b[env_ids] = current_b
        self.hand_traj_step[env_ids] = 0
        self.target_hand_pos_b[env_ids] = current_b
        self.hand_pos_error_b[env_ids] = 0.0

    def _get_current_hand_b(self, env_ids: torch.Tensor):
        """Get current hand positions in body frame relative to torso for given env_ids."""
        hand_pos_w = self.asset.data.body_link_pos_w[env_ids][:, self.hand_body_ids]
        torso_pos_w = self.asset.data.body_link_pos_w[env_ids][:, self.torso_body_id]
        hand_rel_w = hand_pos_w - torso_pos_w.unsqueeze(1)
        quat_w = self.asset.data.root_link_quat_w[env_ids]
        yaw_q = yaw_quat(quat_w)
        left_b = quat_rotate_inverse(yaw_q, hand_rel_w[:, 0])
        right_b = quat_rotate_inverse(yaw_q, hand_rel_w[:, 1])
        return torch.stack([left_b, right_b], dim=1)  # (n, 2, 3)

    def sample_hand_command(self, env_ids: torch.Tensor):
        """Sample a new trajectory: start = current actual position, end = random point."""
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        # Start from current actual hand positions (not previous trajectory end)
        self.hand_traj_start_b[env_ids] = self._get_current_hand_b(env_ids)
        # Sample random end positions in the target range for each hand
        for i in range(2):  # 0=left, 1=right
            rand = torch.rand(n, 3, device=self.device)
            target_b = self.hand_target_min_b + rand * (self.hand_target_max_b - self.hand_target_min_b)
            # Mirror y for left hand (positive y) vs right hand (negative y)
            if i == 1:  # right hand
                target_b[:, 1] = -target_b[:, 1]
            self.hand_traj_end_b[env_ids, i] = target_b
        # Compute per-env duration based on max distance / max_step_size
        diff = self.hand_traj_end_b[env_ids] - self.hand_traj_start_b[env_ids]  # (n, 2, 3)
        max_dist = diff.norm(dim=-1).max(dim=-1).values  # (n,) max over left/right
        duration = (max_dist / self.hand_max_step_size).ceil().long().clamp(min=1)  # (n,)
        self.hand_traj_duration[env_ids] = duration.unsqueeze(-1)
        # Reset step counter
        self.hand_traj_step[env_ids] = 0

    def with_prob(self, n, p):
        return torch.rand(n, device=self.device) < p

    @property
    def command_hand_target(self):
        """Hand position errors in body frame: (N, 6) — [left_x, left_y, left_z, right_x, right_y, right_z]."""
        return self.hand_pos_error_b.reshape(self.num_envs, 6)

    @override
    def debug_draw(self):
        start = self.asset.data.root_link_pos_w + torch.tensor([0.0, 0.0, 0.2], device=self.device)
        yaw_vec = torch.stack(
            [
                self.target_yaw.cos(),
                self.target_yaw.sin(),
                torch.zeros_like(self.target_yaw),
            ],
            1,
        )

        # Compute hand target positions in world frame for visualization
        torso_pos_w = self.asset.data.body_link_pos_w[:, self.torso_body_id]  # (N, 3)
        quat_w = self.asset.data.root_link_quat_w
        yaw_q = yaw_quat(quat_w)
        # Current interpolated target in world frame
        target_left_w = quat_rotate(yaw_q, self.target_hand_pos_b[:, 0]) + torso_pos_w   # (N, 3)
        target_right_w = quat_rotate(yaw_q, self.target_hand_pos_b[:, 1]) + torso_pos_w  # (N, 3)
        # Trajectory start/end in world frame
        traj_start_left_w = quat_rotate(yaw_q, self.hand_traj_start_b[:, 0]) + torso_pos_w
        traj_start_right_w = quat_rotate(yaw_q, self.hand_traj_start_b[:, 1]) + torso_pos_w
        traj_end_left_w = quat_rotate(yaw_q, self.hand_traj_end_b[:, 0]) + torso_pos_w
        traj_end_right_w = quat_rotate(yaw_q, self.hand_traj_end_b[:, 1]) + torso_pos_w
        # Current hand positions in world frame
        hand_pos_w = self.asset.data.body_link_pos_w[:, self.hand_body_ids]  # (N, 2, 3)
        current_left_w = hand_pos_w[:, 0]   # (N, 3)
        current_right_w = hand_pos_w[:, 1]  # (N, 3)

        if self.env.backend == "isaac":
            self.env.debug_draw.vector(
                start,
                self.cmd_linvel_w,
                color=(1.0, 1.0, 1.0, 1.0),
            )
            self.env.debug_draw.vector(
                start,
                yaw_vec,
                color=(0.2, 0.2, 1.0, 1.0),
            )
            # Trajectory start points (yellow)
            self.env.debug_draw.point(
                torch.cat([traj_start_left_w, traj_start_right_w], dim=0),
                color=(1.0, 1.0, 0.0, 1.0),
            )
            # Trajectory end points (magenta)
            self.env.debug_draw.point(
                torch.cat([traj_end_left_w, traj_end_right_w], dim=0),
                color=(1.0, 0.0, 1.0, 1.0),
            )
            # Trajectory lines: start -> end (white)
            self.env.debug_draw.vector(
                torch.cat([traj_start_left_w, traj_start_right_w], dim=0),
                torch.cat([traj_end_left_w - traj_start_left_w, traj_end_right_w - traj_start_right_w], dim=0),
                color=(1.0, 1.0, 1.0, 0.5),
            )
            # Current interpolated target (red)
            self.env.debug_draw.point(
                torch.cat([target_left_w, target_right_w], dim=0),
                color=(1.0, 0.0, 0.0, 1.0),
            )
            # Current hand positions (green)
            self.env.debug_draw.point(
                torch.cat([current_left_w, current_right_w], dim=0),
                color=(0.0, 1.0, 0.0, 1.0),
            )
        elif self.env.backend == "mjlab":
            rpy = torch.zeros(self.num_envs, 3)
            rpy[:, 2] = self.target_yaw.cpu()
            self.axes_handle.batched_wxyzs = quat_from_euler_xyz(rpy)
            self.axes_handle.batched_positions = start.cpu()
            self.lines_handle.points = torch.stack([start, start + self.cmd_linvel_w], 1).cpu()
    
    def symmetry_transform(self):
        # left-right symmetry: flip y velocity and yaw velocity
        transform = symmetry_utils.SymmetryTransform(perm=torch.arange(4), signs=[1, -1, -1, 1])
        return transform

    def hand_target_symmetry_transform(self):
        """Symmetry for hand targets: swap left/right hands and flip y."""
        # command_hand_target is [lx, ly, lz, rx, ry, rz]
        # under left-right symmetry: swap left<->right, flip y
        perm = torch.tensor([3, 4, 5, 0, 1, 2])  # swap left and right
        signs = torch.tensor([1, -1, 1, 1, -1, 1])  # flip y for both
        return symmetry_utils.SymmetryTransform(perm=perm, signs=signs)