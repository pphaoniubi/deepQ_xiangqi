from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        # Convolutional layers to extract spatial features
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 10 * 9, 1024)  # Adjust for board size (10x9)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_size)

    def forward(self, x):
        x = x.view(-1, 1, 10, 9)  # Reshape input to (batch, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Hyperparameters
STATE_SIZE = 90
ACTION_SIZE = 90
BATCH_SIZE = 256
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99997
LEARNING_RATE = 0.001
TARGET_UPDATE = 100

# Replay Buffer
replay_buffer = deque(maxlen=100000)

# Initialize networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

policy_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)  # Move model to device

# Load the checkpoint **only once**
checkpoint_path = r"C:\Users\Peter\Desktop\deepQ_xiangqi\backend\AI\checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load state_dict into the model
policy_net.load_state_dict(checkpoint['policy_net'])

# Ensure model is in evaluation mode
policy_net.eval()

print("Model loaded successfully!")


def generate_moves(board):
    board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = policy_net(board_tensor)  # Forward pass to get Q-values

    # Convert Q-values to a list
    q_values_list = q_values.squeeze().tolist()

    # Find the best action (highest Q-value)
    best_action_index = max(range(len(q_values_list)), key=lambda i: q_values_list[i])

    # Decode the move
    best_move = decode_move(best_action_index)

    # Extract piece at (from_row, from_col)
    from_row, from_col, to_row, to_col = best_move
    piece = board[from_row][from_col]  # Identify which piece is moving

    return {
        "piece": piece, 
        "from": (from_row, from_col), 
        "to": (to_row, to_col)
    }


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