from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from train.lib import Trainer, device

from tqdm.auto import trange, tqdm


class MountainContinuousActor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input):
        return self.net(input)


class Trajectory:
    def __init__(self):
        self.memory = {}

    def add_episode(self, episode_num):
        self.memory[episode_num] = []

    def add_sar(self, episode_num, state, action, reward, other=None):
        self.memory[episode_num].append((state, action, reward, other))

    def clear_all(self):
        self.memory.clear()

    def clear_episode(self, episode_num):
        return self.memory.pop(episode_num)


class ActorCriticTrainer(Trainer):
    def __init__(
        self,
        env: gym.Env,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        device: torch.device | str | None = device,
    ):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.device = device
        self.date = datetime.today().strftime("%Y%m%d-%H%M")
        self.traj = Trajectory()

    def actor_forward(self, **kwargs):
        return self.actor(**kwargs)

    def critic_forward(self, **kwargs):
        return self.critic(**kwargs)

    def actor_update(self):
        pass

    def critic_update(self):
        pass

    def final_update(self):
        pass

    def actor_scheduler_step(self):
        pass

    def critic_scheduler_step(self):
        pass

    def logging(self, episode):
        pass

    def train(
        self,
        num_episodes=500,
        alpha=1e-3,
        alpha_lr_decay=0.5,
        alpha_lr_per_epidsode=200,
        beta: float | None = 1e-3,
        beta_lr_decay: float | None = 0.5,
        beta_lr_per_episode: float | None = 200,
        gamma=0.99,
        per_episodes=1,
        logging_per_episodes=20,
    ):
        env = self.env
        device = self.device
        self.actor.to(device)
        if self.critic is not None:
            self.critic.to(device)
        self.gamma = gamma
        episode_rewards = []
        self.actor_optimizer = Adam(self.actor.parameters(), lr=alpha)
        self.actor_scheduler = StepLR(
            self.actor_optimizer,
            step_size=alpha_lr_per_epidsode,
            gamma=alpha_lr_decay,
        )
        if beta is not None:
            self.critic_optimizer = Adam(self.critic.parameters(), lr=beta)
            self.critic_scheduelr = StepLR(
                self.critic_optimizer,
                step_size=beta_lr_per_episode,
                gamma=beta_lr_decay,
            )
        else:
            self.critic_optimizer = None
            self.critic_scheduler = None

        for e in trange(num_episodes):
            state, _ = env.reset()
            self.traj.add_episode(e)
            self.total_reward = 0
            done = False
            # num_step = 0

            while not done:
                # num_step += 1
                state, done, reward = self.actor_update(state, e)
                self.total_reward += reward
                self.critic_update()
                episode_rewards.append(self.total_reward)

            if e % per_episodes == 0:
                self.final_update()
                self.traj.clear_all()
            self.critic_scheduler_step()
            self.actor_scheduler_step()

            if e % logging_per_episodes == 0:
                self.logging(e)

        env.close()
        return episode_rewards


class REINFORCEBatch(ActorCriticTrainer):
    def actor_forward(self, state):
        state = torch.Tensor(state).to(self.device)
        action_values = self.actor(state)
        dist = Categorical(action_values)
        action = dist.sample()
        return dist, action

    def actor_update(self, state, episode):
        prop, action = self.actor_forward(state)
        np_action = np.expand_dims(action.cpu().numpy(), axis=0)
        next_state, reward, terminated, truncated, info = self.env.step(np_action)
        self.traj.add_sar(episode, state, action, reward, prop)
        return next_state, terminated or truncated, reward

    def final_update(self):
        self.loss = 0
        for e in self.traj.memory.keys():
            v_t = 0
            dist: Categorical
            for s, a, r, dist in self.traj.memory[e][::-1]:
                v_t += r + self.gamma * v_t
                print(
                    f"log prob: {dist.log_prob(a)}, action: {a}, v_t: {v_t}, reward: {r}"
                )
                self.loss += v_t * -dist.log_prob(a)
        self.loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

        # for (s, a, r), p in zip(traj[::-1], props[::-1]):
        # loss += r * torch.log(p)

    def actor_scheduler_step(self):
        self.actor_scheduler.step()

    def logging(self, episode):
        print(
            f"Episode {episode}: total_reward: {self.total_reward}, actor_loss: {self.loss.item()}"
        )
