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
            nn.Linear(hidden_dim, action_dim)
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


def train_linear_q(num_episodes=500, learning_rate=1e-3, gamma=0.99,
                   epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500,
                   render=False):
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
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)
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
            target = reward + gamma * torch.max(next_q_values) * (1 - int(done or truncated))
            loss = (q_value - target.detach()) ** 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
        
        episode_rewards.append(total_reward)
        if episode % 10 == 0:
            print(f"[LinearQ] Episode {episode}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}")
    
    env.close()
    return model, episode_rewards


def train_dqn(num_episodes=500, batch_size=64, learning_rate=1e-3, gamma=0.99,
              buffer_capacity=10000, target_update_freq=10, 
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500,
              render=False):
    """
    DQN (Deep Q-Network)을 이용해 CartPole 에이전트를 학습합니다.
    경험 재생(Replay Buffer)와 타깃 네트워크(Target Network)를 사용합니다.
    
    Returns:
        policy_net: 학습 완료된 정책 네트워크 (QNetwork)
        episode_rewards: 에피소드별 총 보상 리스트
    """
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # 타깃 네트워크는 평가 모드
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer=optimizer, step_size=100, gamma=0.5)
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    episode_rewards = []
    steps_done = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)
            steps_done += 1
            
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            replay_buffer.push(state, action, reward, next_state, done or truncated)
            state = next_state
            
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                current_q = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1, keepdim=True)[0]
                    target_q = rewards + gamma * max_next_q * (1 - dones)
                loss = nn.MSELoss()(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        episode_rewards.append(total_reward)
        scheduler.step()
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if episode % 10 == 0:
            print(f"[DQN] Episode {episode}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}")
    
    env.close()
    return policy_net, episode_rewards


##############################
# 3. 평가 및 렌더링 함수     #
##############################

def evaluate_policy(model, n_episodes=5, render=True):
    # Gymnasium에서는 render_mode="human"로 설정하면 렌더링 모드를 사용할 수 있습니다.
    env = gym.make("CartPole-v1", render_mode="human")
    eval_rewards = []
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            # (렌더링은 env.render()를 호출해도 되지만, render_mode="human"일 경우 자동 표시됩니다.)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = q_values.argmax().item()
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
        eval_rewards.append(total_reward)
        print(f"Evaluation Episode {episode} Reward: {total_reward}")
    env.close()
    return eval_rewards