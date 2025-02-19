import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# Define Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Hyperparameters
gamma = 0.99
lr = 1e-3
batch_size = 64
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
memory_size = 10000
num_episodes = 50
discrete_actions = 11  # Discretizing action space

env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = discrete_actions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_network = QNetwork(state_dim, action_dim).to(device)
target_network = QNetwork(state_dim, action_dim).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

optimizer = optim.Adam(q_network.parameters(), lr=lr)
memory = deque(maxlen=memory_size)


def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = q_network(state)
        return torch.argmax(q_values).item()


def train():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    q_values = q_network(states).gather(1, actions)
    next_q_values = target_network(next_states).max(1, keepdim=True)[0]
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action_idx = select_action(state, epsilon)
        action = np.linspace(-2, 2, discrete_actions)[action_idx]
        next_state, reward, terminated, truncated, _ = env.step([action])
        done = terminated or truncated

        memory.append((state, action_idx, reward, next_state, done))
        train()
        state = next_state
        total_reward += reward

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    target_network.load_state_dict(q_network.state_dict())
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Save trained model
torch.save(q_network.state_dict(), "dqn_pendulum.pth")

env.close()


# Load trained model and evaluate
def evaluate():
    env = gym.make("Pendulum-v1", render_mode="human")
    q_network.load_state_dict(torch.load("dqn_pendulum.pth"))
    q_network.eval()

    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_idx = torch.argmax(q_network(state_tensor)).item()
        action = np.linspace(-2, 2, discrete_actions)[action_idx]
        state, reward, terminated, truncated, _ = env.step([action])
        done = terminated or truncated
        total_reward += reward

    print(f"Evaluation Total Reward: {total_reward}")
    env.close()


evaluate()
