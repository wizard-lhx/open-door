import torch
from typing_extensions import override

from active_adaptation.envs.mdp.base import Reward


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
        error_sq = error.square().sum(dim=-1)  # (N, 2)
        rew = torch.exp(-error_sq / self.sigma).mean(dim=-1, keepdim=True)  # (N, 1)
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
