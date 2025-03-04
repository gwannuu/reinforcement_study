import torch.nn as nn
from train.lib import plot, render, save_video
from train.actor_critic import (
    REINFORCEwithBaseline,
    REINFORCEBatch,
    QValueActorCritic,
)
from train.lib import device

import gymnasium as gym
import matplotlib.pyplot as plt
import torch


class HalfCheetahActor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(17, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6 * 2),
        )

    def forward(self, state):
        mean, log_std = torch.split(self.net(state), 6, dim=-1)
        log_std = torch.clamp(log_std, -20, 3)
        std = torch.exp(log_std)
        return mean, std


class HalfCheetahCritic(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(17, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input):
        output = self.net(input)
        return output


class HalfCheetahQCritic(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(17 + 6, hidden_dim),  # 17 state, 6 action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input):
        output = self.net(input)
        return output


def test_REINFORCE():
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    actor = HalfCheetahActor(hidden_dim=64)
    trainer = REINFORCEBatch(env=env, actor=actor)

    # Test training
    num_episode = 3
    trainer.train(
        num_episodes=num_episode,
        logging_per_episodes=1,
        save_per_episodes=1,
    )
    trainer.plot(*trainer.load_infos())
    frames_list = trainer.render(num_render=2, max_step=500)
    trainer.save_as_video(frames_list, name=num_episode)


def train_REINFORCE():
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    actor = HalfCheetahActor(hidden_dim=64)
    trainer = REINFORCEBatch(env=env, actor=actor)

    trainer.train(
        num_episodes=20000,
        logging_per_episodes=200,
        save_per_episodes=200,
        alpha_lr_per_epidsode=2000,
        beta_lr_per_episode=2000,
    )


def test_REINFORCEwithBaseline():
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    actor = HalfCheetahActor(hidden_dim=64)
    critic = HalfCheetahCritic(hidden_dim=32)
    trainer = REINFORCEwithBaseline(env=env, actor=actor, critic=critic)

    # Test training
    num_episode = 3
    trainer.train(
        num_episodes=num_episode,
        logging_per_episodes=1,
        save_per_episodes=1,
    )
    trainer.plot(*trainer.load_infos())
    frames_list = trainer.render(num_render=2, max_step=500)
    trainer.save_as_video(frames_list, name=num_episode)

def train_REINFORCEwithBaseline():
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    actor = HalfCheetahActor(hidden_dim=64)
    critic = HalfCheetahCritic(hidden_dim=32)
    trainer = REINFORCEwithBaseline(env=env, actor=actor, critic=critic)

    # Test training
    trainer.train(
        num_episodes=20000,
        logging_per_episodes=200,
        save_per_episodes=200,
        alpha_lr_per_epidsode=2000,
        beta_lr_per_episode=2000,
    )

def test_QValueActorCritic():
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    actor = HalfCheetahActor(hidden_dim=64)
    critic = HalfCheetahQCritic(hidden_dim=32)
    trainer = QValueActorCritic(env=env, actor=actor, critic=critic)

    num_episode = 3
    trainer.train(
        num_episodes=num_episode,
        logging_per_episodes=1,
        save_per_episodes=1,
    )
    trainer.plot(*trainer.load_infos())
    frames_list = trainer.render(num_render=2, max_step=500)
    trainer.save_as_video(frames_list, name=num_episode)

def train_QValueActorCritic():
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    actor = HalfCheetahActor(hidden_dim=64)
    critic = HalfCheetahQCritic(hidden_dim=32)
    trainer = QValueActorCritic(env=env, actor=actor, critic=critic)

    trainer.train(
        num_episodes=20000,
        logging_per_episodes=200,
        save_per_episodes=200,
        alpha_lr_per_epidsode=2000,
        beta_lr_per_episode=2000,
    )

def load_trainer_BASELINE(dir, name):
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    actor = HalfCheetahActor(hidden_dim=64)
    # critic = HalfCheetahCritic(hidden_dim=32)
    # trainer = REINFORCEwithBaseline(env=env, actor=actor, critic=critic, dir=dir)
    trainer = REINFORCEBatch(env=env, actor=actor, dir=dir)
    trainer.load_model(name=name)
    return trainer


if __name__ == "__main__":
    train_QValueActorCritic()
    train_REINFORCE()
    train_REINFORCEwithBaseline()
    # test()
    # train()
    # train_reinforce()
    # dir = "HalfCheetah-v5_REINFORCEBatch_20250228-205714"
    # name = "20000"
    # trainer = load_trainer_BASELINE(dir, name)
    # plot(trainer)
    # frames_list = render(trainer)
    # save_video(trainer, frames_list)
