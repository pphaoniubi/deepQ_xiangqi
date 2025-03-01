from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dotenv import load_dotenv
import os
import AI.board_piece
import AI.deepQ


load_dotenv()


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 10 * 9, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_size)

    def forward(self, x):
        x = x.view(-1, 1, 10, 9)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
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
checkpoint_path = os.getenv("FILE_PATH")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load state_dict into the model
policy_net.load_state_dict(checkpoint['policy_net'])

# Ensure model is in evaluation mode
policy_net.eval()

print("Model loaded successfully!")


def generate_moves(board_state):

    checkpoint_path = os.getenv("FILE_PATH")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    
    # Load the trained policy network
    policy_net.load_state_dict(checkpoint['policy_net'])
    policy_net.eval()  # Set model to evaluation mode
    
    print("AI Model Loaded. Generating best move...")

    # Convert board state to tensor
    state_tensor = torch.tensor(board_state, dtype=torch.float32).unsqueeze(0).to(device)

    # Get Q-values for all possible actions
    with torch.no_grad():  # Disable gradient computation for inference
        q_values = policy_net(state_tensor).cpu().numpy().squeeze()

    # Generate list of all legal (piece, action) pairs
    legal_piece_actions = []
    for piece in range(-16, 0):
        legal_moves = AI.board_piece.get_legal_moves(piece, board_state)
        legal_action_indices = AI.deepQ.map_legal_moves_to_actions(legal_moves, ACTION_SIZE) 

        for action in legal_action_indices:
            legal_piece_actions.append((piece, action)) 

    best_q_value = -float('inf')
    best_pair = None
    for piece, action in legal_piece_actions:
        if q_values[action] > best_q_value:
            best_q_value = q_values[action]
            best_pair = (piece, action)

    if best_pair:
        piece, action = best_pair
        (row, col) = AI.deepQ.action_to_2d(action)
        print(f"AI selected piece {piece} with action {(row, col)}")
        return piece, (row, col)
    else:
        print("No valid move found!")
        return None, None


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