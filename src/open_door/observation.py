import torch
from typing_extensions import override

import active_adaptation.utils.symmetry as symmetry_utils
from active_adaptation.envs.mdp.base import Observation


class command_target(Observation):
    """Position error (body frame) + heading error from the command manager."""

    def __init__(self, env):
        super().__init__(env)
        self.command_manager = self.env.command_manager

    @override
    def compute(self):
        return self.command_manager.command_target

    @override
    def symmetry_transform(self):
        # pos_error_b_x (no flip), pos_error_b_y (flip), heading_error (flip)
        return symmetry_utils.SymmetryTransform(
            perm=torch.arange(3), signs=[1, -1, -1]
        )


class command_hand_target(Observation):
    """Hand end-effector position errors in body frame: [left_xyz, right_xyz] (6-dim)."""

    def __init__(self, env):
        super().__init__(env)
        self.command_manager = self.env.command_manager

    @override
    def compute(self):
        return self.command_manager.command_hand_target

    @override
    def symmetry_transform(self):
        return self.command_manager.hand_target_symmetry_transform()
