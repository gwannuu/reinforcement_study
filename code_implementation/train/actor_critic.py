from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
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
            nn.Linear(hidden_dim, 1 * 2),  # mean and std
        )

    def forward(self, input):
        mean, log_std = self.net(input)
        log_std = torch.clamp(log_std, -20, 2)  # prevent extreme gradient value
        std = torch.exp(log_std)
        return mean, std

    def get_action(self, mean, std):
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


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
        critic: torch.nn.Module,
        device: torch.device | str | None = device,
    ):
        super().__init__(env, device, actor=actor, critic=critic)
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
            self.e = e
            self.check_and_save(save_per_episodes=save_per_episodes)

            self.total_reward = 0
            state, _ = env.reset()
            self.traj.add_episode(e)
            self.total_reward = 0
            done = False
            self.num_step = 0
            self.discounted_return = 0

            while not done:
                self.num_step += 1
                state, done, reward = self.actor_update(state, e)
                self.total_reward += reward
                self.critic_update()

            episode_rewards.append(self.total_reward)

            if e % per_episodes == 0:
                self.per_episodes_update()
                self.traj.clear_all()
            self.critic_scheduler_step()
            self.actor_scheduler_step()
            self.check_and_log(logging_per_episodes=logging_per_episodes)

        self.e += 1
        self.check_and_save(save_per_episodes=save_per_episodes)
        self.on_end_callback()
        return episode_rewards


class REINFORCEBatch(ActorCriticTrainer):
    def model_save(self, env_name: str = "", name: str = "latest"):
        model_save_dir = self.get_model_save_dir(env_name)
        Path.mkdir(model_save_dir, exist_ok=True)
        save_path = model_save_dir / f"{name}.pth"
        torch.save(self.actor.state_dict(), save_path)

    def actor_forward(self, state):
        state = torch.Tensor(state).to(self.device)
        mean, std = self.actor(state)
        action, log_prob = self.actor.get_action(mean, std)
        return action, log_prob, mean, std

    def get_np(self, torch_data):
        return np.expand_dims(torch_data.cpu().numpy(), axis=0)

    def actor_update(self, state, episode):
        action, log_prob, _, _ = self.actor_forward(state)
        np_action = self.get_np(action)
        next_state, reward, terminated, truncated, info = self.env.step(np_action)
        self.traj.add_sar(episode, state, action, reward, log_prob)
        return next_state, terminated or truncated, reward

    def per_episodes_update(self):
        self.loss = 0
        count = 0
        for e in self.traj.memory.keys():
            v_t = 0
            for s, a, r, log_prob in self.traj.memory[e][::-1]:
                v_t = r + self.gamma * v_t
                # print(
                #     f"log prob: {log_prob:.3f}, action: {a:.3f}, v_t: {v_t:.3f}, reward: {r:.3f}"
                # )
                self.loss += v_t * -log_prob
                count += 1
        self.loss /= count
        self.loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

    def actor_scheduler_step(self):
        self.actor_scheduler.step()

    def logging(self, episode):
        print(
            f"Episode {episode}: total_reward: {self.total_reward:.3f}, actor_loss: {self.loss.item():.3f}, num_step: {self.num_step}"
        )


class REINFORCEBatchRenderer:
    def __init__(self, env: gym.Env, model: REINFORCEBatch, device=device):
        self.env = env
        self.model = model
        self.model.actor.eval()
        self.device = device
        self.model.actor.to(self.device)

    def rendering(self, num_render=3, max_step=500):
        model = self.model
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
                    a, log_prob, mean, std = model.actor_forward(s)
                    np_a, np_mean, np_std = (
                        model.get_np(a),
                        model.get_np(mean),
                        model.get_np(std),
                    )

                n_s, r, done, truncated, _ = env.step(np_a)
                self.total_reward += r

                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.print_message(frame_bgr, action=np_a, mean=np_mean, std=np_std)
                frames.append(frame_bgr)
                cv2.imshow("Evaluation", frame_bgr)
                if cv2.waitKey(30) & 0xFF in [ord("q"), 27]:
                    done = True
                    break
                s = n_s
                self.step_count += 1
            episodes.append(frames)
        return episodes

    def print_message(self, frame_bgr, **kwargs):
        action, mean, std = kwargs["action"], kwargs["mean"], kwargs["std"]
        messages = []
        messages.append(f"Step: {self.step_count}")
        messages.append(f"Total Reward:{self.total_reward:.3f}")
        messages.append(f"Action: {action[0]:.3f}")
        messages.append(f"mean:{mean[0]:.3f}, std:{std[0]:.3f}")
        h = 20
        for i, m in enumerate(messages):
            color = (0, 0, 255)
            if i == 2 and action[0] >= 0:
                color = (255, 0, 0)
            cv2.putText(
                frame_bgr,
                m,
                (10, h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            h += 30

    def save_as_video(self, frames, dir: str, name: str):
        dir_p = Path(dir)
        path = dir_p / f"{name}.mp4"

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


class REINFORCEwithBaseline(ActorCriticTrainer):
    def model_save(self, env_name: str = "", name: str = "latest"):
        model_save_dir = self.get_model_save_dir(env_name)
        Path.mkdir(model_save_dir, exist_ok=True)
        save_path = model_save_dir / f"{name}.pth"
        torch.save(self.actor.state_dict(), save_path)

    def actor_forward(self, state):
        state = torch.Tensor(state).to(self.device)
        mean, std = self.actor(state)
        action, log_prob = self.actor.get_action(mean, std)
        return action, log_prob, mean, std

    def get_np(self, torch_data):
        return np.expand_dims(torch_data.cpu().numpy(), axis=0)

    def actor_update(self, state, episode):
        action, log_prob, _, _ = self.actor_forward(state)
        np_action = self.get_np(action)
        next_state, reward, terminated, truncated, info = self.env.step(np_action)
        self.discounted_return = reward + self.gamma * self.discounted_return
        self.loss = self.discounted_return * -log_prob
        self.loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        self.loss = 0
        # self.traj.add_sar(episode, state, action, reward, log_prob)
        return next_state, terminated or truncated, reward

    def per_episodes_update(self):
        pass
        # self.loss = 0
        # for e in self.traj.memory.keys():
        #     v_t = 0
        #     for s, a, r, log_prob in self.traj.memory[e][::-1]:
        #         v_t = r + self.gamma * v_t
        #         # print(
        #         #     f"log prob: {log_prob:.3f}, action: {a:.3f}, v_t: {v_t:.3f}, reward: {r:.3f}"
        #         # )
        #         self.loss += v_t * -log_prob
        # self.loss.backward()
        # self.actor_optimizer.step()
        # self.actor_optimizer.zero_grad()

    def actor_scheduler_step(self):
        self.actor_scheduler.step()

    def logging(self, episode):
        print(
            f"Episode {episode}: total_reward: {self.total_reward:.3f}, num_steps: {self.num_step}"
        )
