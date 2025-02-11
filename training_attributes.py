from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
STATE_SIZE = 90  # 10x9 board
ACTION_SIZE = 90  # Simplified action space
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.05  # AI will still explore 5% of the time at the end
EPSILON_DECAY = 0.99997  # Ensures exploration lasts exactly 200,000 episodes
LEARNING_RATE = 0.001
TARGET_UPDATE = 10  # Update target network every 10 episodes

# Replay Buffer
replay_buffer = deque(maxlen=100000)

# Initialize networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()


def train_dqn():
    if len(replay_buffer) < BATCH_SIZE:
        return  # Wait until buffer has enough samples

    # Sample a mini-batch from replay buffer
    batch = random.sample(replay_buffer, BATCH_SIZE)
    
    # Unpack batch: states, actions, rewards, next_states, dones
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    # Compute current Q-values from policy_net
    q_values = policy_net(states).gather(1, actions)  # Select Q-values of chosen actions

    # Compute target Q-values using Bellman equation
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1, keepdim=True)[0]  # Max Q-value of next state
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))  # Q-target

    # Compute loss (Mean Squared Error or Huber Loss)
    loss = F.smooth_l1_loss(q_values, target_q_values)

    # Optimize the policy network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()