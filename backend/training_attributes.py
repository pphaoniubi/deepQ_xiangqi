from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dotenv import load_dotenv
from game_state import game
import os
import board_piece
import time


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
EPSILON_DECAY = 0.99991
LEARNING_RATE = 0.001
TARGET_UPDATE = 500

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


def encode_board_to_1d_board(board):
    board_flat = []
    for row in board:
        for cross in row:
            board_flat.append(cross)

    return np.array(board_flat)

def encode_1d_board_to_board(board_1d):
    if len(board_1d) != 90:
        raise ValueError("Invalid board size: Expected 90 elements")

    board_2d = [board_1d[i * 9:(i + 1) * 9] for i in range(10)]
    return board_2d


def map_legal_moves_to_actions(legal_moves, ACTION_SIZE):
    index = []
    for legal_move in legal_moves:
        index.append(legal_move[1] * 9 + legal_move[0])

    return index

def action_to_2d(action_index):
    row = action_index // 9
    col = action_index % 9 
    return row, col


def step(piece, new_index):
    reward = 0
    board_1d, reward = board_piece.make_move_1d(piece, new_index, encode_board_to_1d_board(game.board), reward)      # make move on 1D

    game.board = encode_1d_board_to_board(board_1d)

    winner = board_piece.is_winning(game.board)
    if winner == "Red wins":
        reward += 2000
        done = True
    elif winner == "Black wins":
        reward -= 2000
        done = True
    elif winner == "Game continues":
        done = False

    return encode_board_to_1d_board(game.board), reward, done

def generate_moves(board_state):

    checkpoint_path = os.getenv("FILE_PATH")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)

    episode = checkpoint['episode']
    print("at episode: ", episode)
    
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
        legal_moves = board_piece.get_legal_moves(piece, board_state)
        legal_action_indices = map_legal_moves_to_actions(legal_moves, ACTION_SIZE) 

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
        (row, col) = action_to_2d(action)
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

    # Convert lists of NumPy arrays into a single NumPy array
    states_np = np.array(states, dtype=np.float32)  # Efficient conversion
    next_states_np = np.array(next_states, dtype=np.float32)
    actions_np = np.array(actions, dtype=np.int64)  # int64 for long tensors
    rewards_np = np.array(rewards, dtype=np.float32).reshape(-1, 1)  # Reshape for batch processing
    dones_np = np.array(dones, dtype=np.float32).reshape(-1, 1)

    # Convert NumPy arrays to PyTorch tensors
    states_tensor = torch.tensor(states_np, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions_np, dtype=torch.long).unsqueeze(1).to(device)
    rewards_tensor = torch.tensor(rewards_np, dtype=torch.float32).to(device)
    next_states_tensor = torch.tensor(next_states_np, dtype=torch.float32).to(device)
    dones_tensor = torch.tensor(dones_np, dtype=torch.float32).to(device)

    # Compute current Q-values from policy_net
    q_values = policy_net(states_tensor).gather(1, actions_tensor)  # Select Q-values of chosen actions

    # Compute target Q-values using Bellman equation
    with torch.no_grad():
        next_q_values = target_net(next_states_tensor).max(1, keepdim=True)[0]  # Max Q-value of next state
        target_q_values = rewards_tensor + (GAMMA * next_q_values * (1 - dones_tensor))  # Q-target

    # Compute loss (Mean Squared Error or Huber Loss)
    loss = F.smooth_l1_loss(q_values, target_q_values)

    # Optimize the policy network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def main():
    global EPSILON

    checkpoint_path = os.getenv("FILE_PATH")
    if os.path.exists(checkpoint_path):

        checkpoint = torch.load("checkpoint.pth")

        print("Saved keys in .pth file:", checkpoint.keys())

        policy_net.load_state_dict(checkpoint['policy_net'])
        target_net.load_state_dict(checkpoint['target_net'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_episode = checkpoint['episode']
        print("starting from: ", start_episode)

    EPISODES = 100000
    start_time = time.time()

    try:
        for episode in range(start_episode, EPISODES):

            game.board = game.board_init
            state = encode_board_to_1d_board(game.board)
            total_reward = 0
            turn = 1
            
            count = 0
            for t in range(200):

                legal_piece_actions = []
                for piece in range(1, 17) if turn == 1 else range(-16, 0):
                    legal_moves = board_piece.get_legal_moves(piece, game.board)
                    legal_action_indices = map_legal_moves_to_actions(legal_moves, ACTION_SIZE) 

                    for action in legal_action_indices:
                        legal_piece_actions.append((piece, action))


                if random.random() < EPSILON:
                    piece, action = random.choice(legal_piece_actions)

                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor).cpu().detach().numpy().squeeze()

                    # Find the best (piece, action) pair using Q-values
                    best_q_value = -float('inf')
                    best_pair = None
                    for piece, action in legal_piece_actions:
                        if q_values[action] > best_q_value:
                            best_q_value = q_values[action]
                            best_pair = (piece, action)

                    piece, action = best_pair  # Select the best piece-action pair

                # Take the action and observe the new state
                next_state, reward, done = step(piece, action) 
                replay_buffer.append((state, action, reward, next_state, done))

                # Train the network
                train_dqn()                         

                state = next_state
                total_reward += reward

                turn = 1 - turn

                count = t

                if done:
                    break

            # Decay epsilon
            if EPSILON > EPSILON_MIN:
                EPSILON *= EPSILON_DECAY

            # Update target network periodically
            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
                checkpoint = {
                    'policy_net': policy_net.state_dict(),
                    'target_net': target_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'episode': episode
                }
                
                torch.save(checkpoint, "checkpoint.pth")
                print(f"Checkpoint saved at episode {episode}")


            print(f"Episode {episode}, Total Reward: {total_reward}, Move count: {count}")

    except KeyboardInterrupt:
        end_time = time.time()  # Stop timer on Ctrl+C
        running_time = end_time - start_time
        print("\nRunning time:", running_time, "seconds")


main()