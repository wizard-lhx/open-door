import torch
import wandb
from typing import List, TYPE_CHECKING
from typing_extensions import override

from active_adaptation.envs.mdp.base import Reward

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor


class hand_tracking_exp(Reward):
    """Exponential reward for tracking hand end-effector target positions.
    
    Uses the hand_pos_error_b from the command manager (body frame).
    reward = exp(-||error||^2 / sigma)
    """

    def __init__(self, env, weight: float, sigma: float = 0.1):
        super().__init__(env, weight)
        self.sigma = sigma

    @override
    def compute(self) -> torch.Tensor:
        # hand_pos_error_b: (N, 2, 3)
        error = self.command_manager.hand_pos_error_b
        error_sq = error.square().sum(dim=-1)   # (N, 2)
        rew = torch.exp(-error_sq / self.sigma).mean(dim=-1, keepdim=True)  # (N, 1)

        # Log error statistics for tuning sigma
        error_flat = error_sq.detach().cpu().reshape(-1)
        self.env.extra["hand_track/error_mean"] = error_flat.mean()

        return rew


class hand_tracking_l2(Reward):
    """Negative L2 penalty for hand end-effector tracking error.
    
    reward = -||error||^2  (averaged over both hands)
    """

    def __init__(self, env, weight: float):
        super().__init__(env, weight)

    @override
    def compute(self) -> torch.Tensor:
        error = self.command_manager.hand_pos_error_b
        error_sq = error.square().sum(dim=-1)  # (N, 2)
        return -error_sq.mean(dim=-1, keepdim=True)  # (N, 1)


class foot_clearance(Reward):
    """Reward the swinging feet for clearing a specified height off the ground.

    Encourages the robot to lift its feet to a target height when swinging,
    weighted by the foot's lateral velocity (only active when foot is moving).
    """

    def __init__(self, env, body_names: str, target_height: float, std: float = 0.05,
                 tanh_mult: float = 2.0, weight: float = 1.0):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.target_height = target_height
        self.std = std
        self.tanh_mult = tanh_mult

    @override
    def compute(self) -> torch.Tensor:
        foot_z = self.asset.data.body_link_pos_w[:, self.body_ids, 2]
        foot_z_target_error = (foot_z - self.target_height).square()
        foot_vel_xy = self.asset.data.body_lin_vel_w[:, self.body_ids, :2].norm(dim=-1)
        foot_velocity_tanh = torch.tanh(self.tanh_mult * foot_vel_xy)
        reward = foot_z_target_error * foot_velocity_tanh
        return torch.exp(-reward.sum(dim=1, keepdim=True) / self.std)


class feet_gait(Reward):
    """Reward a desired periodic gait pattern for bipeds.

    Encourages the robot to follow a periodic gait where each foot alternates
    between contact and air phases according to the specified period and offset.

    Args:
        body_names: Regex pattern for foot body names.
        period: Gait cycle period in seconds.
        offset: Phase offset for each foot (0.0 = in-phase, 0.5 = anti-phase).
        threshold: Fraction of half-period above which the mode time is considered correct.
    """
    supported_backends = ("isaac", "mjlab")

    def __init__(self, env, body_names: str, period: float, offset: List[float],
                 threshold: float = 0.55, weight: float = 1.0):
        super().__init__(env, weight)
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.contact_body_ids = self.contact_sensor.find_bodies(body_names)[0]
        self.contact_body_ids = torch.tensor(self.contact_body_ids, device=self.device)
        self.period = period
        self.offset = torch.tensor(offset, device=self.device, dtype=torch.float)
        self.threshold = threshold

    @override
    def compute(self) -> torch.Tensor:
        contact_time = self.contact_sensor.data.current_contact_time[:, self.contact_body_ids]
        air_time = self.contact_sensor.data.current_air_time[:, self.contact_body_ids]
        in_contact = contact_time > 0.0

        # current phase for each foot based on episode time
        t = self.env.episode_length_buf.unsqueeze(1).float() * self.env.step_dt
        phase = ((t / self.period) + self.offset.unsqueeze(0)) % 1.0  # [N, num_feet]

        # desired: stance phase [0, 0.5), swing phase [0.5, 1.0)
        desired_contact = phase < 0.5

        in_mode_time = torch.where(in_contact, contact_time, air_time)
        correct_mode = in_contact == desired_contact
        half_period = self.period / 2.0
        time_reward = (in_mode_time / half_period).clamp_max(1.0)
        reward = (correct_mode.float() * time_reward).mean(dim=1)

        active = ~self.command_manager.is_standing_env
        return reward.reshape(self.num_envs, 1), active.reshape(self.num_envs, 1)


class pos_tracking_exp(Reward):
    """Exponential reward for root position tracking (body frame error).

    reward = exp(-||pos_error_b||^2 / sigma)
    """

    def __init__(self, env, weight: float, sigma: float = 0.5):
        super().__init__(env, weight)
        self.sigma = sigma

    @override
    def compute(self) -> torch.Tensor:
        pos_error = self.command_manager.pos_error_b  # (N, 2)
        error_sq = pos_error.square().sum(dim=-1, keepdim=True)  # (N, 1)
        rew = torch.exp(-error_sq / self.sigma)
        self.env.extra["pos_track/error_sq_dist"] = error_sq.detach().cpu().reshape(-1).numpy()
        return rew


class heading_tracking_exp(Reward):
    """Exponential reward for heading tracking.

    reward = exp(-heading_error^2 / sigma)
    """

    def __init__(self, env, weight: float, sigma: float = 0.25):
        super().__init__(env, weight)
        self.sigma = sigma

    @override
    def compute(self) -> torch.Tensor:
        heading_error = self.command_manager.heading_error  # (N, 1)
        error_sq = heading_error.square()
        rew = torch.exp(-error_sq / self.sigma)
        self.env.extra["heading_track/error_sq_dist"] = error_sq.detach().cpu().reshape(-1).numpy()
        return rew
