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
import piece_move

load_dotenv()

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()

        # Initial Convolutional Block
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual Blocks (3 Blocks)
        self.res1 = self._residual_block(64)
        self.res2 = self._residual_block(64)
        self.res3 = self._residual_block(64)

        # Policy Head (Predicts best moves)
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 10 * 9, action_size)

        # Value Head (Predicts game outcome)
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(10 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def _residual_block(self, channels):
        """Creates a simple Residual Block"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # print("Input shape before reshape:", x.shape)
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, 10, 9)  # Reshape input

        # Initial Conv Block
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual Blocks
        x = F.relu(self.res1(x) + x)
        x = F.relu(self.res2(x) + x)
        x = F.relu(self.res3(x) + x)

        # Policy Head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = F.softmax(self.policy_fc(policy), dim=1)  # Probability distribution over moves

        # Value Head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Outputs between -1 and 1 (win/loss)

        return policy, value

STATE_SIZE = 90
ACTION_SIZE = 90
BATCH_SIZE = 256
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99991
LEARNING_RATE = 0.001
TARGET_UPDATE = 500


red_replay_buffer = deque(maxlen=100000)
black_replay_buffer = deque(maxlen=100000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

red_policy_net = DQN(ACTION_SIZE).to(device)
red_target_net = DQN(ACTION_SIZE).to(device)
red_target_net.load_state_dict(red_policy_net.state_dict())
red_optimizer = optim.Adam(red_policy_net.parameters(), lr=LEARNING_RATE)

black_policy_net = DQN(ACTION_SIZE).to(device)
black_target_net = DQN(ACTION_SIZE).to(device)
black_target_net.load_state_dict(black_policy_net.state_dict())
black_optimizer = optim.Adam(black_policy_net.parameters(), lr=LEARNING_RATE)

red_checkpoint_path = os.getenv("RED_FILE_PATH")
black_checkpoint_path = os.getenv("BLACK_FILE_PATH")

if os.path.exists(red_checkpoint_path):
    red_checkpoint = torch.load(red_checkpoint_path, map_location=device)
    red_policy_net.load_state_dict(red_checkpoint['policy_net'])
    red_policy_net.eval()
    print("Red model loaded successfully!")

if os.path.exists(black_checkpoint_path):
    black_checkpoint = torch.load(black_checkpoint_path, map_location=device)
    black_policy_net.load_state_dict(black_checkpoint['policy_net'])
    black_policy_net.eval()
    print("Black model loaded successfully!")


def action_to_2d(action_index):
    row = action_index // 9
    col = action_index % 9 
    return row, col



def generate_moves(board_state, turn):
    if turn == 1:
        checkpoint_path = os.getenv("RED_FILE_PATH")
        policy_net = red_policy_net
    else:
        checkpoint_path = os.getenv("BLACK_FILE_PATH")
        policy_net = black_policy_net
    print(checkpoint_path)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    episode = checkpoint['episode']
    print(f"Using {'red' if turn == 1 else 'black'} agent at episode: {episode}")
    
    # Load the trained policy network
    policy_net.load_state_dict(checkpoint['policy_net'])
    policy_net.eval()
    
    print("AI Model Loaded. Generating best move...")

    # Convert board state to tensor
    state_tensor = torch.tensor(board_state, dtype=torch.float32).unsqueeze(0).to(device)

    # Get Q-values for all possible actions
    with torch.no_grad():
        policy_output, _ = policy_net(state_tensor)  # Unpack the tuple (policy, value)
        q_values = policy_output.cpu().numpy().squeeze()

    # Generate list of all legal (piece, action) pairs
    piece_range = range(1, 17) if turn == 1 else range(-16, 0)
    legal_piece_actions = []
    for piece in piece_range:
        board_np = np.array(board_state, dtype=np.int32)
        legal_moves = piece_move.get_legal_moves(piece, board_np)
        legal_action_indices = piece_move.map_legal_moves_to_actions(legal_moves) 
        for action in legal_action_indices:
            legal_piece_actions.append((piece, action))

    if not legal_piece_actions:
        print("No valid moves found!")
        return None, None

    best_q_value = -float('inf')
    best_pair = None
    for piece, action in legal_piece_actions:
        if q_values[action] > best_q_value:
            best_q_value = q_values[action]
            best_pair = (piece, action)

    if best_pair:
        piece, action = best_pair
        row, col = action_to_2d(action)
        print(f"AI selected piece {piece} with action {(row, col)}")
        return piece, (row, col)
    else:
        print("No valid move found!")
        return None, None


def train_dqn(turn):
    """Train the appropriate network based on the current turn."""
    if turn == 1:  # Red's turn
        if len(red_replay_buffer) < BATCH_SIZE:
            return
        
        batch = random.sample(red_replay_buffer, BATCH_SIZE)
        policy_net = red_policy_net
        target_net = red_target_net
        optimizer = red_optimizer
        
    else:  # Black's turn
        if len(black_replay_buffer) < BATCH_SIZE:
            return
            
        batch = random.sample(black_replay_buffer, BATCH_SIZE)
        policy_net = black_policy_net
        target_net = black_target_net
        optimizer = black_optimizer

    # Unpack batch
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to tensors
    states_np = np.array(states, dtype=np.float32)
    next_states_np = np.array(next_states, dtype=np.float32)
    actions_np = np.array(actions, dtype=np.int64)
    rewards_np = np.array(rewards, dtype=np.float32).reshape(-1, 1)
    dones_np = np.array(dones, dtype=np.float32).reshape(-1, 1)

    # Move to device
    states_tensor = torch.tensor(states_np, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions_np, dtype=torch.long).unsqueeze(1).to(device)
    rewards_tensor = torch.tensor(rewards_np, dtype=torch.float32).to(device)
    next_states_tensor = torch.tensor(next_states_np, dtype=torch.float32).to(device)
    dones_tensor = torch.tensor(dones_np, dtype=torch.float32).to(device)

    # Compute current Q values
    policy_output, _ = policy_net(states_tensor)  # Unpack the tuple (policy, value)
    current_q_values = policy_output.gather(1, actions_tensor)

    # Compute next Q values
    with torch.no_grad():
        next_policy_output, _ = target_net(next_states_tensor)  # Unpack the tuple
        next_q_values = next_policy_output.max(1, keepdim=True)[0]
        target_q_values = rewards_tensor + (GAMMA * next_q_values * (1 - dones_tensor))

    # Compute loss and optimize
    loss = F.smooth_l1_loss(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def main():
    global EPSILON

    # Load checkpoints if they exist
    if os.path.exists(red_checkpoint_path) and os.path.exists(black_checkpoint_path):
        red_checkpoint = torch.load(red_checkpoint_path)
        black_checkpoint = torch.load(black_checkpoint_path)

        red_policy_net.load_state_dict(red_checkpoint['policy_net'])
        red_target_net.load_state_dict(red_checkpoint['target_net'])
        red_optimizer.load_state_dict(red_checkpoint['optimizer'])

        black_policy_net.load_state_dict(black_checkpoint['policy_net'])
        black_target_net.load_state_dict(black_checkpoint['target_net'])
        black_optimizer.load_state_dict(black_checkpoint['optimizer'])

        start_episode = max(red_checkpoint['episode'], black_checkpoint['episode'])

        print(f"Starting from episode: {start_episode}")
    else: 
        start_episode = 0

    EPISODES = 200001
    start_time = time.time()

    try:
        for episode in range(start_episode, EPISODES):
            game.board = game.board_init
            state = piece_move.encode_board_to_1d_board(game.board)
            total_red_reward = 0
            total_black_reward = 0
            red_count = 0
            black_count = 0
            turn = 1
            red_move_history = []
            black_move_history = []
            for t in range(160):
                current_policy_net = red_policy_net if turn == 1 else black_policy_net
                
                board_np = np.array(game.board, dtype=np.int32)
                legal_piece_actions = piece_move.generate_all_legal_actions(
                        turn,
                        board_np,
                        piece_move.get_legal_moves,
                        piece_move.map_legal_moves_to_actions

                )

                if not legal_piece_actions:
                    break

                if random.random() < EPSILON:
                    piece, action = random.choice(legal_piece_actions)
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        policy_output, _ = current_policy_net(state_tensor)  # Unpack the tuple
                        q_values = policy_output.cpu().numpy().squeeze() 

                    # Find the best legal move
                    best_q_value = float('-inf')
                    best_pair = None
                    for piece, action in legal_piece_actions:
                        if q_values[action] > best_q_value:
                            best_q_value = q_values[action]
                            best_pair = (piece, action)
                    piece, action = best_pair
                
                if turn == 1:
                    red_count += 1
                    if len(red_move_history) < 4:
                        red_move_history.append((piece, action))
                    elif len(red_move_history) == 4:
                        red_move_history.pop(0)
                        red_move_history.append((piece, action))
                else: 
                    black_count += 1
                    if len(black_move_history) < 4:
                        black_move_history.append((piece, action))
                    elif len(black_move_history) == 4:
                        black_move_history.pop(0)
                        black_move_history.append((piece, action))

                # Take action and observe next state
                move_history = red_move_history if turn == 1 else black_move_history
                count = red_count if turn == 1 else black_count
                next_state, reward, done = step(piece, action, turn, move_history, count)

                # Store transition in appropriate replay buffer
                if turn == 1:
                    red_replay_buffer.append((state, action, reward, next_state, done))
                    total_red_reward += reward
                else:
                    black_replay_buffer.append((state, action, reward, next_state, done))
                    total_black_reward += reward

                # Train the current agent
                train_dqn(turn)

                # Update state and turn
                state = next_state
                turn = 1 - turn  # Switch turns
                game_count = t
                if done:
                    break

            # Decay epsilon
            if EPSILON > EPSILON_MIN:
                EPSILON *= EPSILON_DECAY

            # Update target networks periodically
            if episode % TARGET_UPDATE == 0:
                red_target_net.load_state_dict(red_policy_net.state_dict())
                black_target_net.load_state_dict(black_policy_net.state_dict())
                
                # Save checkpoints
                red_checkpoint = {
                    'policy_net': red_policy_net.state_dict(),
                    'target_net': red_target_net.state_dict(),
                    'optimizer': red_optimizer.state_dict(),
                    'episode': episode
                }
                
                black_checkpoint = {
                    'policy_net': black_policy_net.state_dict(),
                    'target_net': black_target_net.state_dict(),
                    'optimizer': black_optimizer.state_dict(),
                    'episode': episode
                }
                
                torch.save(red_checkpoint, "red_checkpoint.pth")
                torch.save(black_checkpoint, "black_checkpoint.pth")
                print(f"Checkpoints saved at episode {episode}")

            print(f"Episode {episode}, Red Reward: {total_red_reward}, Black Reward: {total_black_reward}, Move count: {game_count}")

    except KeyboardInterrupt:
        end_time = time.time()
        running_time = end_time - start_time
        print("\nRunning time:", running_time, "seconds")


main()

# pip install numpy python-dotenv FastAPi pymysql uvicorn cryptography
# python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
# uvicorn main_api:app --reload