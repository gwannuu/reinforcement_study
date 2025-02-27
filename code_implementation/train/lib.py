# lib.py
import abc
import random
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import trange

# device = "mps"
device= "cuda:1"


def get_np(torch_data):
    if torch_data is not None:
        return torch_data.cpu().numpy()


def add_text(prefix, array: np.ndarray):
    t = prefix
    for i in np.squeeze(array):
        t += f"{i:.3f} "
    return t


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class Trainer(ABC):
    def __init__(
        self,
        env: gym.Env,
        device: torch.device | str | None = device,
        dir: str | Path = None,
        **kwargs,
    ):
        self.env = env
        self.device = device
        self.dir = dir
        self.date = datetime.today().strftime("%Y%m%d-%H%M")
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def move_to_device(self):
        pass

    def get_model_save_dir(self, env_name: str = "", info: str = "") -> Path:
        if self.dir is not None:
            return self.dir
        if env_name == "":
            env_name = self.env.env.spec.id
        if info == "":
            info = self.__class__.__name__
        save_path = Path.cwd() / f"{env_name}_{info}_{self.date}"
        return save_path

    @abstractmethod
    def model_save(self):
        pass


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


class ActorCriticTrainer(Trainer, ABC):
    def __init__(
        self,
        env: gym.Env,
        actor: torch.nn.Module,
        critic: torch.nn.Module = None,
        device: torch.device | str | None = device,
        dir: str | Path | None = None,
    ):
        super().__init__(env, device, dir, actor=actor, critic=critic)
        self.date = datetime.today().strftime("%Y%m%d-%H%M")
        self.traj = Trajectory()

    def move_to_device(self):
        self.actor.to(self.device)
        if self.critic is not None:
            self.critic.to(self.device)

    @abstractmethod
    def actor_get_action(self):
        pass

    def actor_forward(self, **kwargs):
        return self.actor(**kwargs)

    def critic_forward(self, **kwargs):
        if self.critic is not None:
            return self.critic(**kwargs)

    def actor_update(self):
        pass

    def critic_update(self):
        pass

    def per_episodes_update(self):
        pass

    def actor_scheduler_step(self):
        pass

    def critic_scheduler_step(self):
        pass

    @abstractmethod
    def logging(self):
        pass

    def on_start_callback(self):
        self.e = 0
        Path(self.get_model_save_dir()).mkdir(exist_ok=True)

    def on_start_episode_callback(self):
        pass

    def on_end_callback(self):
        self.e = 0
        self.env.close()

    def check_and_save(self, save_per_episodes):
        if self.e % save_per_episodes == 0 and save_per_episodes != -1:
            self.model_save(name=f"{self.e}")

    def check_and_log(self, logging_per_episodes):
        if self.e % logging_per_episodes == 0 and logging_per_episodes != -1:
            self.logging(self.e)

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
        save_per_episodes=50,
    ):
        self.on_start_callback()
        params = {k: v for k, v in locals().items() if k != "self"}
        with open(f"{self.get_model_save_dir()}/train_params.txt", "w") as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")

        env = self.env
        self.move_to_device()
        self.gamma = gamma
        episode_rewards = []
        self.actor_optimizer = Adam(self.actor.parameters(), lr=alpha)
        self.actor_scheduler = StepLR(
            self.actor_optimizer,
            step_size=alpha_lr_per_epidsode,
            gamma=alpha_lr_decay,
        )
        if beta is not None and self.critic is not None:
            self.critic_optimizer = Adam(self.critic.parameters(), lr=beta)
            self.critic_scheduler = StepLR(
                self.critic_optimizer,
                step_size=beta_lr_per_episode,
                gamma=beta_lr_decay,
            )
        else:
            self.critic_optimizer = None
            self.critic_scheduler = None

        for e in trange(num_episodes):
            self.on_start_episode_callback()
            self.e = e
            self.check_and_save(save_per_episodes=save_per_episodes)

            self.total_reward = 0
            state, _ = env.reset()
            self.traj.add_episode(e)
            done = False
            self.num_step = 0
            self.discounted_return = 0

            while not done:
                self.num_step += 1
                np_action = self.actor_forward(state)
                next_state, reward, terminated, truncated, info = self.env.step(
                    np.squeeze(np_action)
                )
                self.traj.add_sar(e, state, np_action, reward, self.log_prob.clone())
                state = next_state
                done = terminated or truncated

                self.total_reward += reward

            if e % per_episodes == 0:
                self.per_episodes_update()
                self.traj.clear_all()
            self.check_and_log(logging_per_episodes=logging_per_episodes)
            episode_rewards.append(self.total_reward)
            self.critic_scheduler_step()
            self.actor_scheduler_step()

        self.e += 1
        self.check_and_log(logging_per_episodes=logging_per_episodes)
        self.check_and_save(save_per_episodes=save_per_episodes)
        self.on_end_callback()
        returns = self.get_return()
        return episode_rewards, returns

    def load_model(self, name):
        self.actor.load_state_dict(
            torch.load(f"{self.dir}/actor_{name}.pth", map_location=self.device)
        )
        if self.critic is not None:
            self.critic.load_state_dict(
                torch.load(f"{self.dir}/critic_{name}.pth", map_location=self.device)
            )

    def save_as_video(self, frames_list, name: str, dir: str = None):
        if dir is None:
            dir_p = Path(self.get_model_save_dir())
        else:
            dir_p = Path(dir)

        dir_p.mkdir(exist_ok=True)

        for i, frames in enumerate(frames_list):
            path = dir_p / f"{name}_{i}.mp4"

            if len(frames) > 0:
                frames = np.array(frames)
                height, width, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(
                    f"{path}",
                    fourcc,
                    30.0,
                    (width, height),
                )

                for frame_bgr in frames:
                    out.write(frame_bgr)
                out.release()
                print(f"video is saved in {path}")

    def render(self, num_render=3, max_step=500, wait_time=30):
        self.move_to_device()
        env = self.env
        episodes = []
        for n in range(num_render):
            s, _ = env.reset()
            self.total_reward = 0
            self.step_count = 0
            done = False
            frames = []

            while not done and self.step_count < max_step:
                with torch.no_grad():
                    a = self.actor_forward(state=s)
                    _ = get_np(self.critic_forward(state=s))
                n_s, r, done, truncated, _ = env.step(np.squeeze(a))
                self.total_reward += r

                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.print_message(frame_bgr)
                frames.append(frame_bgr)

                s = n_s
                self.step_count += 1
            for f in frames:
                cv2.imshow("Evaluation", f)
                if cv2.waitKey(wait_time) & 0xFF in [ord("q"), 27]:
                    done = True
                    break
            episodes.append(frames)
        return episodes
