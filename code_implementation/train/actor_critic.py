from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from train.lib import (
    ActorCriticTrainer,
    compare_plot,
    get_np,
    add_text,
    figsize,
    convert_to_log_values,
    plot_list,
)


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
            nn.Linear(hidden_dim, 1),  # mean and std
        )

    def forward(self, state):
        return self.net(state)


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


class REINFORCEBatch(ActorCriticTrainer):
    def model_save(self, env_name: str = "", name: str = "latest"):
        model_save_dir = self.get_model_save_dir(env_name)
        Path.mkdir(model_save_dir, exist_ok=True)
        save_path = model_save_dir / f"actor_{name}.pth"
        torch.save(self.actor.state_dict(), save_path)

    def actor_get_action(self, mean, std):
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, mean, std

    def actor_forward(self, state):
        state = torch.Tensor(state).to(device=self.device, dtype=torch.float32)
        mean, std = self.actor(state)
        action, self.log_prob, self.mean, self.std = self.actor_get_action(
            mean=mean,
            std=std,
        )
        np_action = get_np(action)
        self.np_action = np_action
        return np_action

    def per_episodes_update(self):
        self.actor_loss = 0
        for e in self.traj.memory.keys():
            v_t = 0
            for s, a, r, log_prob in self.traj.memory[e][::-1]:
                v_t = r + self.gamma * v_t
                # print(
                #     f"log prob: {log_prob:.3f}, action: {a:.3f}, v_t: {v_t:.3f}, reward: {r:.3f}"
                # )
                self.actor_loss += torch.mean(v_t * -log_prob)
        self.actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

    def actor_scheduler_step(self):
        self.actor_scheduler.step()

    def logging(self, episode):
        print(
            f"Episode {episode}: total_reward: {self.total_reward:.3f}, actor_object: {self.actor_loss.item():.3f}, num_step: {self.num_step}"
        )

    def load_infos(self):
        total_losses = self.load_all_lines_from_txt("total_reward")
        actor_objects = self.load_all_lines_from_txt("actor_object")
        actor_lr_list = self.load_all_lines_from_txt("actor_lr")
        return total_losses, actor_objects, actor_lr_list

    def get_return(self):
        return self.actor_loss

    def print_message(self, frame_bgr):
        action = self.np_action
        mean = get_np(self.mean)
        std = get_np(self.std)
        # critic_value = np.squeeze(get_np(self.critic_value))
        messages = []
        messages.append(f"Step: {self.step_count}")
        messages.append(f"Total Reward:{self.total_reward:.3f}")
        messages.append(add_text(prefix="Action: ", array=action))
        messages.append(add_text(prefix="Mean: ", array=mean))
        messages.append(add_text(prefix="Std: ", array=std))
        # messages.append(f"critic_value: {critic_value:.3f}")
        h = 20
        for i, m in enumerate(messages):
            cv2.putText(
                frame_bgr,
                m,
                (10, h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            h += 30

    def plot(self, rewards_list, actor_objects, actor_lr_list, jupyter=False):
        save_dir = self.get_model_save_dir()
        plot_list(
            rewards_list,
            title="REINFORCE(baseline) Training Rewards",
            xlabel="Episode",
            ylabel="Total Reward",
            save_path=None if jupyter else f"{save_dir}/total_reward.png",
        )
        plot_list(
            actor_objects,
            title="Actor Loss Trends",
            xlabel="Episode",
            ylabel="Actor Object",
            save_path=None if jupyter else f"{save_dir}/loss.png",
        )
        plot_list(
            actor_lr_list,
            title="Actor LR",
            xlabel="Episode",
            ylabel="learning rate",
            save_path=None if jupyter else f"{save_dir}/actor_lr.png",
        )


class REINFORCEwithBaseline(ActorCriticTrainer):
    def on_start_callback(self):
        super().on_start_callback()
        self.actor_losses = []
        self.critic_losses = []

    def on_start_episode_callback(self):
        self.total_reward = 0

    def model_save(self, env_name: str = "", name: str = "latest"):
        model_save_dir = self.get_model_save_dir(env_name)
        Path.mkdir(model_save_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), model_save_dir / f"actor_{name}.pth")
        torch.save(self.critic.state_dict(), model_save_dir / f"critic_{name}.pth")

    def actor_get_action(self, mean, std):
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, mean, std

    def actor_forward(self, state):
        state = torch.Tensor(state).to(device=self.device, dtype=torch.float32)
        mean, std = self.actor(state)
        action, self.log_prob, self.mean, self.std = self.actor_get_action(
            mean=mean,
            std=std,
        )
        np_action = get_np(action)
        self.np_action = np_action
        return np_action

    def critic_forward(self, state, action):
        state = torch.from_numpy(state).to(device=self.device, dtype=torch.float32)
        self.critic_value = self.critic(state)
        return self.critic_value

    def per_episodes_update(self):
        self.critic_loss = 0
        self.actor_loss = 0
        for e in self.traj.memory.keys():
            v_t = 0
            v_t_list = []
            state_list = []
            log_probs = []
            for s, a, r, log_prob in self.traj.memory[e][::-1]:
                v_t = np.float32(r) + self.gamma * v_t
                v_t_list.insert(0, v_t)
                state_list.insert(0, s)
                log_probs.insert(0, log_prob)
                # print(
                #     f"log prob: {log_prob:.3f}, action: {a:.3f}, v_t: {v_t:.3f}, reward: {r:.3f}"
                # )
            v_ts = torch.tensor(v_t_list)[:, None].to(self.device)
            states = np.array(state_list)
            v_estimated = self.critic_forward(states, action=None)
            self.critic_loss += F.mse_loss(v_estimated, v_ts, reduction="mean")

            with torch.no_grad():
                v_estimated = self.critic_forward(states, action=None)
            advantage = v_ts - v_estimated
            log_probs = torch.stack(log_probs).to(self.device)
            # print(
            #     f"advantage shape: {advantage.shape}, v_ts shape: {v_ts.shape},log_probs shape: {log_probs.shape}"
            # )
            self.actor_loss += torch.mean(advantage * -log_probs)
            self.actor_losses.append(self.actor_loss.item())

        self.actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

    def actor_scheduler_step(self):
        self.actor_scheduler.step()

    def critic_scheduler_step(self):
        self.critic_scheduler.step()

    def logging(self, episode):
        print(
            f"Episode {episode}: total_reward: {self.total_reward:.3f}, actor_object: {self.actor_loss.item():.3f}, critic_loss: {self.critic_loss.item():.3f}, num_step: {self.num_step}"
        )

    def load_infos(self):
        total_losses = self.load_all_lines_from_txt("total_reward")
        actor_objects = self.load_all_lines_from_txt("actor_object")
        critic_losses = self.load_all_lines_from_txt("critic_loss")
        actor_lr_list = self.load_all_lines_from_txt("actor_lr")
        critic_lr_list = self.load_all_lines_from_txt("critic_lr")
        return total_losses, actor_objects, critic_losses, actor_lr_list, critic_lr_list

    def plot(
        self,
        rewards_list,
        actor_objects,
        critic_losses,
        actor_lr_list,
        critic_lr_list,
        jupyter=False,
    ):
        save_dir = self.get_model_save_dir()
        plot_list(
            rewards_list,
            title="REINFORCE(baseline) Training Rewards",
            xlabel="Episode",
            ylabel="Total Reward",
            save_path=None if jupyter else f"{save_dir}/total_reward.png",
        )
        compare_plot(
            actor_objects,
            critic_losses,
            title="Actor and Critic Loss Trends",
            xlabel="Episode",
            y1label="Actor Object",
            y2label="Critic Loss",
            save_path=None if jupyter else f"{save_dir}/loss.png",
        )
        plot_list(
            actor_lr_list,
            title="Actor LR",
            xlabel="Episode",
            ylabel="learning rate",
            save_path=None if jupyter else f"{save_dir}/actor_lr.png",
        )
        plot_list(
            critic_lr_list,
            title="Critic LR",
            xlabel="Episode",
            ylabel="learning rate",
            save_path=None if jupyter else f"{save_dir}/critic_lr.png",
        )

    def get_return(self):
        return self.actor_losses, self.critic_losses

    def print_message(self, frame_bgr):
        action = self.np_action
        mean = get_np(self.mean)
        std = get_np(self.std)
        critic_value = np.squeeze(get_np(self.critic_value))
        messages = []
        messages.append(f"Step: {self.step_count}")
        messages.append(f"Total Reward:{self.total_reward:.3f}")
        messages.append(add_text(prefix="Action: ", array=action))
        messages.append(add_text(prefix="Mean: ", array=mean))
        messages.append(add_text(prefix="Std: ", array=std))
        messages.append(f"critic_value: {critic_value:.3f}")
        h = 20
        for i, m in enumerate(messages):
            cv2.putText(
                frame_bgr,
                m,
                (10, h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            h += 30
