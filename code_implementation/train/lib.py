# lib.py
import abc
import random
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

# device = "mps"
device= "cuda:1"


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
        **kwargs,
    ):
        self.env = env
        self.device = device
        self.date = datetime.today().strftime("%Y%m%d-%H%M")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_model_save_dir(self, env_name: str = "", info: str = "") -> Path:
        if env_name == "":
            env_name = self.env.env.spec.id
        if info == "":
            info = self.__class__.__name__
        save_path = Path.cwd() / f"{env_name}_{info}_{self.date}"
        return save_path

    @abstractmethod
    def model_save(self):
        pass
