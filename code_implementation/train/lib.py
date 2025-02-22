# lib.py
import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn.functional as F
import abc
from pathlib import Path
from datetime import datetime


device = "mps"


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


class Trainer:
    def __init__(
        self,
        env: gym.Env,
        policy_net: torch.nn.Module,
        target_net: torch.nn.Module,
        device: torch.device | str | None = device,
    ):
        self.env = env
        self.policy_net = policy_net
        self.target_net = target_net
        self.device = device
        self.date = datetime.today().strftime("%Y%m%d-%H%M")

    def get_model_save_dir(self, env_name: str = "") -> Path:
        if env_name == "":
            env_name = self.env.env.spec.id
        save_path = Path.cwd() / f"{env_name}_{self.date}"
        return save_path

    def model_save(self, env_name: str = "", name: str = "latest"):
        model_save_dir = self.get_model_save_dir(env_name)
        Path.mkdir(model_save_dir, exist_ok=True)
        save_path = model_save_dir / f"{name}.pth"
        torch.save(self.policy_net.state_dict(), save_path)
