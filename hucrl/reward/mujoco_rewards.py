"""Rewards of Mujoco Environments."""

from abc import ABCMeta

import numpy as np
import torch
from rllib.model import AbstractModel
from rllib.util.utilities import get_backend


class MujocoReward(AbstractModel, metaclass=ABCMeta):
    """Base class for mujoco rewards."""

    def __init__(self, action_cost=0.01, sparse=False, goal=None):
        super().__init__(
            goal=goal, dim_state=(), dim_action=(self.dim_action,), model_kind="rewards"
        )
        self.action_scale = 1
        self.action_cost = action_cost
        self.sparse = sparse
        self.reward_ctrl = torch.tensor(0.0)
        self.reward_state = torch.tensor(0.0)

    def action_reward(self, action):
        """Get action reward."""
        action = action[..., : self.dim_action[0]]  # get only true dimensions.
        bk = get_backend(action)
        if self.sparse:
            return bk.exp(-bk.square(action / self.action_scale).sum(-1)) - 1
        else:
            return -bk.square(action).sum(-1)

    def get_reward(self, reward_state, reward_control):
        """Get reward distribution from reward_state, reward_control tuple."""
        self.reward_ctrl = self.action_cost * reward_control
        self.reward_state = reward_state
        reward = self.reward_state + self.reward_ctrl
        try:
            return reward.float().unsqueeze(-1), torch.zeros(1)
        except AttributeError:
            return reward, torch.zeros(1)


class CartPoleReward(MujocoReward):
    """Reward of MBRL CartPole Environment."""

    dim_action = 1

    def __init__(self, action_cost=0.01, pendulum_length=0.6):
        super().__init__(action_cost)
        self.length = pendulum_length

    def forward(self, state, action, next_state):
        """See `AbstractReward.forward()'."""
        bk = get_backend(next_state)
        end_effector = self._get_ee_pos(next_state[..., 0], next_state[..., 1])

        reward_state = bk.exp(-bk.square(end_effector).sum(-1) / (self.length ** 2))
        return self.get_reward(reward_state, self.action_reward(action))

    def _get_ee_pos(self, x0, theta):
        bk = get_backend(x0)
        sin, cos = bk.sin(theta), bk.cos(theta)
        return bk.stack([x0 - self.length * sin, -self.length * (1 + cos)], -1)


class PilcoCartPoleReward(MujocoReward):
    """Reward of Pilco CartPole Environment."""

    dim_action = 1

    def __init__(self, pendulum_length=0.6):
        super().__init__(0.)
        self.length = pendulum_length

    def forward(self, state, action, next_state):
        """See `AbstractReward.forward()'."""
        bk = get_backend(next_state)
        position = self._get_ee_pos(next_state[..., 0], next_state[..., 1])
        goal = bk.Tensor([0.0, self.length])
        squared_distance = bk.sum((position - goal) ** 2, axis=-1)
        squared_sigma = 0.25 ** 2
        costs = 1 - bk.exp(-0.5 * squared_distance / squared_sigma)

        return self.get_reward(-costs, self.action_reward(action))

    def _get_ee_pos(self, x0, theta):
        bk = get_backend(x0)
        pole_x = self.length * bk.sin(theta)
        pole_y = self.length * bk.cos(theta)
        return bk.stack([x0 + pole_x, pole_y]).T


class HalfCheetahReward(MujocoReward):
    """Reward of MBRL HalfCheetah Environment."""

    dim_action = 6

    def __init__(self, action_cost=0.1):
        super().__init__(action_cost)
        self.dt = 0.05

    def forward(self, state, action, next_state):
        """See `AbstractReward.forward()'."""
        reward_state = (next_state[..., 0] - state[..., 0]) / self.dt
        return self.get_reward(reward_state, self.action_reward(action))


class HalfCheetahV2Reward(MujocoReward):
    """Reward of MBRL HalfCheetah V2 Environment."""

    dim_action = 6

    def __init__(self, action_cost=0.1):
        super().__init__(action_cost)

    def forward(self, state, action, next_state):
        """See `AbstractReward.forward()'."""
        reward_state = next_state[..., 0]
        return self.get_reward(reward_state, self.action_reward(action))


class PusherReward(MujocoReward):
    """Reward of MBRL Pusher Environment."""

    dim_action = 7

    def __init__(self, action_cost=0.1, goal=torch.tensor([0.45, -0.05, -0.323])):
        # goal[-1] = -0.275
        super().__init__(action_cost, goal=goal)

    def forward(self, state, action, next_state):
        """See `AbstractReward.forward()'."""
        bk = get_backend(state)
        if bk == np:
            goal = np.array(self.goal)
        else:
            goal = self.goal

        if isinstance(state, torch.Tensor):
            state = state.detach()
        tip_pos = self.get_ee_position(state)
        obj_pos = state[..., -3:]

        dist_to_obj = obj_pos - tip_pos
        dist_to_goal = obj_pos - goal

        reward_dist_to_obj = -bk.abs(dist_to_obj).sum(-1)
        reward_dist_to_goal = -bk.abs(dist_to_goal)[..., :-1].sum(-1)
        reward_state = 1.25 * reward_dist_to_goal + 0.5 * reward_dist_to_obj

        self.reward_dist_to_obj = 0.5 * reward_dist_to_obj
        self.reward_dist_to_goal = 1.25 * reward_dist_to_goal

        return self.get_reward(reward_state, self.action_reward(action))

    @staticmethod
    def get_ee_position(state):
        """Get the end effector position."""
        bk = get_backend(state)
        theta1, theta2 = state[..., 0], state[..., 1]
        theta3, theta4 = state[..., 2:3], state[..., 3:4]

        rot_axis = bk.stack(
            [
                bk.cos(theta2) * bk.cos(theta1),
                bk.cos(theta2) * bk.sin(theta1),
                -bk.sin(theta2),
            ],
            -1,
        )
        rot_perp_axis = bk.stack(
            [-bk.sin(theta1), bk.cos(theta1), bk.zeros_like(theta1)], -1
        )

        cur_end = bk.stack(
            [
                0.1 * bk.cos(theta1) + 0.4 * bk.cos(theta1) * bk.cos(theta2),
                0.1 * bk.sin(theta1) + 0.4 * bk.sin(theta1) * bk.cos(theta2) - 0.6,
                -0.4 * bk.sin(theta2),
            ],
            -1,
        )

        for length, hinge, roll in [(0.321, theta4, theta3)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            if bk is torch:
                perp_all_axis = torch.tensor(
                    perp_all_axis, dtype=torch.get_default_dtype()
                )

            x = rot_axis * bk.cos(hinge)
            y = bk.sin(hinge) * bk.sin(roll) * rot_perp_axis
            z = -bk.sin(hinge) * bk.cos(roll) * perp_all_axis

            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            if bk is torch:
                new_rot_perp_axis = torch.tensor(
                    new_rot_perp_axis, dtype=torch.get_default_dtype()
                )

            norm = bk.sqrt(bk.square(new_rot_axis).sum(-1))
            new_rot_perp_axis[norm < 1e-30] = rot_perp_axis[norm < 1e-30]

            new_rot_perp_axis /= bk.sqrt(bk.square(new_rot_perp_axis).sum(-1))[
                ..., None
            ]

            rot_axis, rot_perp_axis = new_rot_axis, new_rot_perp_axis
            cur_end = cur_end + length * new_rot_axis

        return cur_end


class BARLReacherReward(MujocoReward):

    dim_action = 2

    def __init__(self):
        super().__init__(0)

    def forward(self, state, action, next_state):

        bk = get_backend(state)
        vec = next_state[..., -2:]
        reward_dist = -bk.norm(vec, dim=-1)
        action = action[..., :2]
        reward_ctrl = -bk.sum(bk.square(action), dim=-1)
        reward = reward_dist + reward_ctrl
        return self.get_reward(reward, self.action_reward(action))


class PendulumReward(MujocoReward):
    dim_action = 1
    def __init__(self):
        super().__init__(0)

    @staticmethod
    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def forward(self, state, action, next_state):
        bk = get_backend(state)
        th = next_state[..., 0]
        thdot = next_state[..., 1]
        action = action[..., 0]
        costs = self.angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (action ** 2)
        return self.get_reward(-costs, self.action_reward(action))


class BetaTrackingReward(MujocoReward):
    dim_action = 1
    BETA_IDX = 0
    def __init__(self, target=2.):
        self.target = target
        super().__init__(0)

    def forward(self, state, action, next_state):
        bk = get_backend(state)
        betas = next_state[..., BETA_IDX]
        iqr = 0.8255070447921753
        median = 1.622602
        betas = betas * iqr + median
        return self.get_reward(-1 * bk.abs(betas - target), self.action_reward(action))


class PlasmaTrackingReward(MujocoReward):
    dim_action = 2
    def __init__(self):
        self.targets = [0.4544037912481128, 0.515012974224002]
        self.idxes = [0, 2]
        super().__init__(0)

    def forward(self, state, action, next_state):
        bk = get_backend(state)
        signals = next_state[..., idxes]
        rew = -1 * bk.sum(bk.abs(signals - self.targets), axis=-1)
        return self.get_reward(rew, self.action_reward(action))


class ReacherReward(MujocoReward):
    """Reward of Reacher Environment."""

    dim_action = 7

    def __init__(self, action_cost=0.01, sparse=False, goal=None):
        super().__init__(action_cost, sparse, goal=goal)
        self.action_scale = 2.0
        self.length_scale = 0.45

    def forward(self, state, action, next_state):
        """See `AbstractReward.forward()'."""
        bk = get_backend(state)
        with torch.no_grad():
            # goal = state[..., -3:]
            dist_to_target = self.get_ee_position(next_state) - self.goal

        if self.sparse:
            reward_state = bk.exp(
                -bk.square(dist_to_target).sum(-1) / (self.length_scale ** 2)
            )
        else:
            reward_state = -bk.square(dist_to_target).sum(-1)

        return self.get_reward(reward_state, self.action_reward(action))

    @staticmethod
    def get_ee_position(state):
        """Get the end effector position."""
        bk = get_backend(state)
        theta1, theta2 = state[..., 0], state[..., 1]
        theta3, theta4 = state[..., 2:3], state[..., 3:4]
        theta5, theta6 = state[..., 4:5], state[..., 5:6]

        rot_axis = bk.stack(
            [
                bk.cos(theta2) * bk.cos(theta1),
                bk.cos(theta2) * bk.sin(theta1),
                -bk.sin(theta2),
            ],
            -1,
        )
        rot_perp_axis = bk.stack(
            [-bk.sin(theta1), bk.cos(theta1), bk.zeros_like(theta1)], -1
        )

        cur_end = bk.stack(
            [
                0.1 * bk.cos(theta1) + 0.4 * bk.cos(theta1) * bk.cos(theta2),
                0.1 * bk.sin(theta1) + 0.4 * bk.sin(theta1) * bk.cos(theta2) - 0.188,
                -0.4 * bk.sin(theta2),
            ],
            -1,
        )
        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            if bk is torch:
                perp_all_axis = torch.tensor(
                    perp_all_axis, dtype=torch.get_default_dtype()
                )

            x = rot_axis * bk.cos(hinge)
            y = bk.sin(hinge) * bk.sin(roll) * rot_perp_axis
            z = -bk.sin(hinge) * bk.cos(roll) * perp_all_axis

            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            if bk is torch:
                new_rot_perp_axis = torch.tensor(
                    new_rot_perp_axis, dtype=torch.get_default_dtype()
                )

            norm = bk.sqrt(bk.square(new_rot_perp_axis).sum(-1))
            new_rot_perp_axis[norm < 1e-30] = rot_perp_axis[norm < 1e-30]

            new_rot_perp_axis /= bk.sqrt(bk.square(new_rot_perp_axis).sum(-1))[
                ..., None
            ]

            rot_axis, rot_perp_axis = new_rot_axis, new_rot_perp_axis
            cur_end = cur_end + length * new_rot_axis

        return cur_end

barl_reward_models = {
        'pilcocartpole-v0': PilcoCartPoleReward,
        'bacreacher-v0': BARLReacherReward,
        'betatracking-v0': BetaTrackingReward,
        'bacpendulum-v0': PendulumReward,
        'plasmatracking-v0': PlasmaTrackingReward,
        }
