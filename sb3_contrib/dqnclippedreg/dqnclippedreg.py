from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy

from stable_baselines3.common import logger
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule


class DQNClippedReg(DQN):
    """
    DQNClippedReg adds functions DQNClipped and DQNReg from paper: https://arxiv.org/abs/2101.03958
    Additional hyperparameters
    :param loss_type: Use standard Huber loss or custom loss function (0: standard , 2: DQN Clipped, 3: DQN Reg)
    :param dqn_reg_loss_weight: If using dqn_reg_loss, weight regularization to use. Defaults to 0.1
    """

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        loss_type: int = 0,
        dqn_reg_loss_weight: float = 0.1,
        _init_setup_model: bool = True,
    ):

        super(DQNClippedReg, self).__init__(
            policy,
            env,
            DQNPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Discrete,),
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None
        # parameters for DQN Clipped and DQN Reg Losses
        self.loss_type = loss_type # 1 for DQN Clipped Loss, 2 for DQN Reg Loss, else standard huber loss
        self.dqn_reg_loss_weight = dqn_reg_loss_weight
        self.DQN_CLIPPED_LOSS_CONSTANT = 2
        self.DQN_REG_LOSS_CONSTANT = 3

        if _init_setup_model:
            self._setup_model()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            if self.loss_type == self.DQN_CLIPPED_LOSS_CONSTANT:
                loss = self.dqn_clipped_loss(current_q_values, target_q_values, next_q_values, self.gamma)
            elif self.loss_type == self.DQN_REG_LOSS_CONSTANT:
                loss = self.dqn_reg_loss(current_q_values, target_q_values, self.dqn_reg_loss_weight)
            else:
                # standard DQN
                # Compute Huber loss (less sensitive to outliers)
                loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))

    def dqn_clipped_loss(self, current_q, target_q, target_q_a_max, gamma):
        """
        Custom loss function per paper: https://arxiv.org/abs/2101.03958
        In DQN, replaces Huber/MSE loss between train and target network
        :param current_q: Q(st, at) of training network
        :param target_q: Max Q value from the target network, including the reward and gamma. r + gamma * Q_target(st+1,a)
        :param target_q_a_max: Q_target(st+1, a) where a is the action of the max Q value
        :param gamma: the discount factor
        """
        # loss = max[current_q, delta^2 + target_q] + max[delta,gamma*(target_q_a_max)^2]
        delta = current_q - target_q
        left_max = th.max(current_q, th.pow(delta, 2) + target_q)
        right_max = th.max(delta, gamma * th.pow(target_q_a_max, 2))
        loss = th.mean(left_max + right_max)

        return loss

    def dqn_reg_loss(self, current_q, target_q, weight=0.1):
        """
        Custom loss function per paper: https://arxiv.org/abs/2101.03958
        In DQN, replaces Huber/MSE loss between train and target network
        :param current_q: Q(st, at) of training network
        :param target_q: Max Q value from the target network, including the reward and gamma. r + gamma * Q_target(st+1,a)
        :param weight: scalar. weighted term that regularizes Q value. Paper defaults to 0.1 but theorizes that tuning this
        per env to some positive value may be beneficial.
        """
        # loss = weight * Q(st, at) + delta^2
        delta = current_q - target_q
        loss = th.mean(weight * current_q + th.pow(delta, 2))

        return loss
