# lib.py
import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import abc
from pathlib import Path
from datetime import datetime
import cv2

device = "mps"


class LinearQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(LinearQNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, action_dim)

    def forward(self, x):
        return self.fc(x)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


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


def train_linear_q(
    num_episodes=500,
    learning_rate=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=500,
    render=False,
):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = LinearQNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    episode_rewards = []
    steps_done = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if render:
                env.render()
            # ε-탐욕 정책
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
                -1.0 * steps_done / epsilon_decay
            )
            steps_done += 1

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            # TD(0) 업데이트
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            next_q_values = model(next_state_tensor)
            q_value = q_values[0, action]
            target = reward + gamma * torch.max(next_q_values) * (
                1 - int(done or truncated)
            )
            loss = (q_value - target.detach()) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        episode_rewards.append(total_reward)
        if episode % 10 == 0:
            print(
                f"[LinearQ] Episode {episode}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}"
            )

    env.close()
    return model, episode_rewards


class Trainer:
    def __init__(
        self,
        env: gym.Env,
        policy_net: torch.nn.Module,
        target_net: torch.nn.Module,
        device: torch.device | str,
    ):
        self.env = env
        self.policy_net = policy_net
        self.target_net = target_net
        self.device = device
        self.date = datetime.today().strftime("%Y%m%d-%H%M")

    def model_save(self, name: str):
        cur = Path.cwd()
        model_save_dir = cur / self.date
        Path.mkdir(model_save_dir, exist_ok=True)
        save_path = model_save_dir / f"{name}.pth"
        torch.save(self.policy_net.state_dict(), save_path)


class DQN_Trainer(Trainer):
    def dqn(
        self,
        num_episodes=500,
        batch_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_capacity=10000,
        target_update_freq=10,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
        render=False,
        step_lr_decay=0.8,
        step_lr_episode=100,
        print_freq=10,
        model_save_freq=50,
    ):
        env = self.env
        device = self.device
        policy_net = self.policy_net.to(device)
        target_net = self.target_net.to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        scheduler = StepLR(
            optimizer=optimizer, step_size=step_lr_episode, gamma=step_lr_decay
        )
        replay_buffer = ReplayBuffer(buffer_capacity)

        episode_rewards = []
        last_loss = 0

        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            num_step = 0

            while not done:
                if render:
                    env.render()
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
                    -1.0 * episode / epsilon_decay
                )

                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                num_step += 1

                if len(replay_buffer) > batch_size:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(
                        batch_size
                    )
                    states = torch.FloatTensor(states).to(device)
                    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                    next_states = torch.FloatTensor(next_states).to(device)
                    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                    current_q = policy_net(states).gather(1, actions)
                    with torch.no_grad():
                        max_next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
                        target_q = rewards + gamma * max_next_q * (1 - dones)
                    loss = nn.MSELoss()(current_q, target_q)
                    last_loss = loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            if episode % model_save_freq == 0:
                self.model_save(name=f"{episode}")
            episode_rewards.append(total_reward)
            scheduler.step()
            if episode % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if episode % print_freq == 0:
                print(
                    f"[DQN] Episode {episode}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}, loss: {last_loss:.3f}"
                )
        env.reset()
        self.model_save(f"{num_episodes}")
        return episode_rewards


##############################
# 3. 평가 및 렌더링 함수     #
##############################


def evaluate_policy(model, n_episodes=5, max_step=500, save_dir="", model_path=""):
    """
    학습된 모델(LinearQNetwork 또는 QNetwork)을 평가합니다.
    - 환경은 "rgb_array" 모드로 생성되어, OpenCV로 프레임을 받아옵니다.
    - 각 프레임의 좌상단에 현재 step, 총 보상(total reward), 그리고
      행동 선택 직전의 before_action (여기서는 선택된 행동)을 표시합니다.
    """
    # device 설정 (모델과 동일한 device 사용)
    # device = torch.device("" if torch.cuda.is_available() else "cpu")
    device = "mps"

    # rgb_array 모드로 환경 생성 (OpenCV로 직접 프레임을 처리)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    eval_rewards = []
    model.eval()
    model.to(device)

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        frames = []

        while not done and step_count < 500:
            # 모델 입력을 위해 state를 tensor로 변환
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            # 행동 선택 (여기서 before_action은 선택된 행동을 의미)
            action = q_values.argmax().item()
            before_action = action  # 선택된 행동

            # 환경에서 한 스텝 진행
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1

            # 프레임을 rgb_array 모드로 가져옴
            frame = env.render()  # frame: (H, W, 3) RGB 형식 numpy array
            # OpenCV는 BGR 순서를 사용하므로 변환
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 좌상단에 텍스트 오버레이: Step, Total Reward, Action
            cv2.putText(
                frame_bgr,
                f"Step: {step_count}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame_bgr,
                f"Total Reward: {total_reward}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame_bgr,
                f"Action Value: {q_values[0][0]:.3f}, {q_values[0][1]:.3f}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            # 프레임 출력 (ESC나 'q' 키를 누르면 종료)
            frames.append(frame_bgr)
            cv2.imshow("Evaluation", frame_bgr)
            if cv2.waitKey(30) & 0xFF in [ord("q"), 27]:
                done = True
                break

            state = next_state

        eval_rewards.append(total_reward)
        print(f"Evaluation Episode {episode} Reward: {total_reward}")

        if len(frames) > 0:
            frames = np.array(frames)
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4 코덱
            out = cv2.VideoWriter(
                f"{save_dir}/{model_path}_eval_{episode}.mp4",
                fourcc,
                30.0,
                (width, height),
            )

            # ----- (2) 한 프레임씩 동영상으로 쓰기 -----
            for frame_bgr in frames:
                out.write(frame_bgr)

            # ----- (3) 자원 해제 -----
            out.release()
            print(f"video is saved in {save_dir}/eval_{episode}.mp4")

    env.close()
    cv2.destroyAllWindows()
    return eval_rewards
