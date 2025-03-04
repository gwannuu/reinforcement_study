from train.actor_critic import (
    REINFORCEwithBaseline,
    REINFORCEBatch,
    MountainContinuousActorV2,
    MountainContinuousCritic,
)
from train.lib import device

import gymnasium as gym
import matplotlib.pyplot as plt
import torch


def test():
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


def train():
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


def train_reinforce():
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    actor = MountainContinuousActorV2(hidden_dim=64)
    trainer = REINFORCEBatch(env=env, actor=actor)
    trainer.train(
        num_episodes=20000,
        logging_per_episodes=500,
        save_per_episodes=500,
        alpha_lr_per_epidsode=2000,
        beta_lr_per_episode=2000,
    )


def load_trainer(dir, name):
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    actor = MountainContinuousActorV2(hidden_dim=32)
    critic = MountainContinuousCritic(hidden_dim=32)
    trainer = REINFORCEwithBaseline(env=env, actor=actor, critic=critic, dir=dir)
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
    # test()
    train()
    train_reinforce()
    # dir = "MountainCarContinuous-v0_REINFORCEwithBaseline_20250228-174219"
    # name = "2000"
    # trainer = load_trainer(dir, name)
    # plot(trainer)
    # frames_list = render(trainer)
    # save_video(trainer, frames_list)
