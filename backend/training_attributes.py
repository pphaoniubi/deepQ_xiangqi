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
from utils import encode_board_to_1d_board, encode_1d_board_to_board


load_dotenv()


class DQN(nn.Module):
    def __init__(self, action_size):
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


def map_legal_moves_to_actions(legal_moves):
    index = []
    for legal_move in legal_moves:
        index.append(legal_move[1] * 9 + legal_move[0])
    return index

def action_to_2d(action_index):
    row = action_index // 9
    col = action_index % 9 
    return row, col


def step(piece, new_index, turn, move_history, count):
    if turn == 1:    
        # Exponential penalty based on move count
        if count > 30:  # Start penalty earlier
            count_penalty = -10 * (2 ** ((count - 30) / 20))  # Exponential penalty
        else: 
            count_penalty = 0

        board_1d, reward_red = board_piece.make_move_1d(piece, new_index, encode_board_to_1d_board(game.board), turn, move_history=move_history)      # make move on 1D
        reward_red += count_penalty

        game.board = encode_1d_board_to_board(board_1d)

        winner = board_piece.is_winning(game.board)
        if winner == "Red wins":
            done = True
            reward_red += 2000  # Bigger reward for winning
        elif winner == "Game continues":
            done = False

        return encode_board_to_1d_board(game.board), reward_red, done
    
    elif turn == 0:    
        # Exponential penalty based on move count
        if count > 30:  # Start penalty earlier
            count_penalty = -10 * (2 ** ((count - 30) / 20))  # Exponential penalty
        else: 
            count_penalty = 0

        board_1d, reward_black = board_piece.make_move_1d(piece, new_index, encode_board_to_1d_board(game.board), turn, move_history=move_history)      # make move on 1D
        reward_black += count_penalty

        game.board = encode_1d_board_to_board(board_1d)

        winner = board_piece.is_winning(game.board)
        if winner == "Black wins":
            done = True
            reward_black += 2000  # Bigger reward for winning
        elif winner == "Game continues":
            done = False

        return encode_board_to_1d_board(game.board), reward_black, done

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
        q_values = policy_net(state_tensor).cpu().numpy().squeeze()

    # Generate list of all legal (piece, action) pairs
    piece_range = range(1, 17) if turn == 1 else range(-16, 0)
    legal_piece_actions = []
    for piece in piece_range:
        legal_moves = board_piece.get_legal_moves(piece, board_state)
        legal_action_indices = map_legal_moves_to_actions(legal_moves) 
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
    current_q_values = policy_net(states_tensor).gather(1, actions_tensor)

    # Compute next Q values
    with torch.no_grad():
        next_q_values = target_net(next_states_tensor).max(1, keepdim=True)[0]
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

    EPISODES = 1000000
    start_time = time.time()

    try:
        for episode in range(start_episode, EPISODES):
            game.board = game.board_init
            state = encode_board_to_1d_board(game.board)
            total_red_reward = 0
            total_black_reward = 0
            red_count = 0
            black_count = 0
            turn = 1
            red_move_history = []
            black_move_history = []
            for t in range(200):
                current_policy_net = red_policy_net if turn == 1 else black_policy_net
                piece_range = range(1, 17) if turn == 1 else range(-16, 0)


                legal_piece_actions = []
                for piece in piece_range:
                    legal_moves = board_piece.get_legal_moves(piece, game.board)
                    legal_action_indices = map_legal_moves_to_actions(legal_moves) 
                    for action in legal_action_indices:
                        legal_piece_actions.append((piece, action))

                if not legal_piece_actions:
                    break

                if random.random() < EPSILON:
                    piece, action = random.choice(legal_piece_actions)
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_values = current_policy_net(state_tensor).cpu().numpy().squeeze()

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

# pip install numpy torch python-dotenv FastAPi pymysql uvicorn cryptography
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
