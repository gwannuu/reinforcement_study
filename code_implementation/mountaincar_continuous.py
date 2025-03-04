import torch.nn as nn

from train.actor_critic import REINFORCEwithBaseline, REINFORCEBatch, QValueActorCritic
from train.lib import device

import gymnasium as gym
import matplotlib.pyplot as plt
import torch


class MountainContinuousActor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 * 2),  # mean and std
        )

    def forward(self, input):
        mean, log_std = self.net(input)
        log_std = torch.clamp(log_std, -2, 0.5)  # prevent extreme gradient value
        std = torch.exp(log_std)
        return mean, std


class MountainContinuousActorV2(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 * 2),  # mean and std
        )

    def forward(self, input):
        mean, log_std = self.net(input)
        log_std = torch.clamp(log_std, -2, 0.5)  # prevent extreme gradient value
        std = torch.exp(log_std)
        return mean, std


class MountainContinuousCritic(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)


class MountainContinuousQCritic(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input):
        return self.net(input)


def test_REINFORCE():
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    actor = MountainContinuousActorV2(hidden_dim=64)
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
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    actor = MountainContinuousActorV2(hidden_dim=32)
    trainer = REINFORCEBatch(env=env, actor=actor)
    trainer.train(
        num_episodes=20000,
        logging_per_episodes=500,
        save_per_episodes=500,
        alpha_lr_per_epidsode=2000,
        beta_lr_per_episode=2000,
    )

def test_REINFORCEwithBaseline():
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    actor = MountainContinuousActorV2(hidden_dim=64)
    critic = MountainContinuousCritic(hidden_dim=32)
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
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    actor = MountainContinuousActorV2(hidden_dim=32)
    critic = MountainContinuousCritic(hidden_dim=32)
    trainer = REINFORCEwithBaseline(env=env, actor=actor, critic=critic)

    trainer.train(
        num_episodes=20000,
        logging_per_episodes=500,
        save_per_episodes=500,
        alpha_lr_per_epidsode=2000,
        beta_lr_per_episode=2000,
    )


def test_QValueActorCritic():
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    actor = MountainContinuousActorV2(hidden_dim=64)
    critic = MountainContinuousQCritic(hidden_dim=32)
    trainer = QValueActorCritic(env=env, actor=actor, critic=critic)

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


def train_QValueActorCritic():
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    actor = MountainContinuousActorV2(hidden_dim=32)
    critic = MountainContinuousQCritic(hidden_dim=32)
    trainer = QValueActorCritic(env=env, actor=actor, critic=critic)

    trainer.train(
        num_episodes=20000,
        logging_per_episodes=500,
        save_per_episodes=500,
        alpha_lr_per_epidsode=2000,
        beta_lr_per_episode=2000,
    )


def load_trainer(dir, name):
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    actor = MountainContinuousActorV2(hidden_dim=64)
    # critic = MountainContinuousCritic(hidden_dim=64)
    trainer = REINFORCEBatch(env=env, actor=actor, critic=None, dir=dir)
    trainer.load_model(name=name)
    return trainer


def plot(trainer):
    trainer.plot(*trainer.load_infos())


def render(trainer):
    frames_list = trainer.render()
    return frames_list


def save_video(trainer, frames_list):
    trainer.save_as_video(frames_list=frames_list)


if __name__ == "__main__":
    train_QValueActorCritic()
    train_REINFORCE()
    train_REINFORCEwithBaseline()
    # test()
    # train()
    # train_reinforce()
    # dir = "MountainCarContinuous-v0_REINFORCEBatch_20250301-063634"
    # name = "20000"
    # trainer = load_trainer(dir, name)
    # plot(trainer)
    # frames_list = render(trainer)
    # save_video(trainer, frames_list)
