from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from dotenv import load_dotenv
from game_state import game
import os
import time
import piece_move
from concurrent.futures import ThreadPoolExecutor
from collections import deque

load_dotenv()

class XiangqiNet(nn.Module):
    def __init__(self, action_size):
        super(XiangqiNet, self).__init__()

        # Initial Convolutional Block
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # 15 Residual Blocks
        self.res_blocks = nn.ModuleList([
            self._residual_block(128) for _ in range(15)
        ])

        # Policy Head
        self.policy_conv1 = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_bn1 = nn.BatchNorm2d(32)
        self.policy_conv2 = nn.Conv2d(32, 8, kernel_size=1)
        self.policy_bn2 = nn.BatchNorm2d(8)
        self.policy_fc = nn.Linear(8 * 10 * 9, action_size)

        # Value Head
        self.value_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 10 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def _residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 10, 9)

        # Initial Conv
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual Tower
        for res_block in self.res_blocks:
            x = F.relu(res_block(x) + x)

        # Policy Head
        policy = F.relu(self.policy_bn1(self.policy_conv1(x)))
        policy = F.relu(self.policy_bn2(self.policy_conv2(policy)))
        policy = policy.view(batch_size, -1)
        policy = F.softmax(self.policy_fc(policy), dim=1)

        # Value Head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(batch_size, -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

STATE_SIZE = 90
ACTION_SIZE = 90
BATCH_SIZE = 256
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99991
LEARNING_RATE = 0.001
TARGET_UPDATE = 1000

red_replay_buffer = deque(maxlen=500000)
black_replay_buffer = deque(maxlen=500000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = GradScaler("cuda")

red_policy_net = XiangqiNet(ACTION_SIZE).to(device)
red_target_net = XiangqiNet(ACTION_SIZE).to(device)
red_target_net.load_state_dict(red_policy_net.state_dict())
red_optimizer = optim.Adam(red_policy_net.parameters(), lr=LEARNING_RATE)

black_policy_net = XiangqiNet(ACTION_SIZE).to(device)
black_target_net = XiangqiNet(ACTION_SIZE).to(device)
black_target_net.load_state_dict(black_policy_net.state_dict())
black_optimizer = optim.Adam(black_policy_net.parameters(), lr=LEARNING_RATE)

red_checkpoint_path = os.getenv("RED_FILE_PATH")
black_checkpoint_path = os.getenv("BLACK_FILE_PATH")

def convert_if_memoryview(x):
    if "_memoryviewslice" in str(type(x)) or isinstance(x, memoryview):
        return np.array(x, dtype=np.int32, copy=True)
    return x

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
    board_state = piece_move.encode_board_to_1d_board(board_state)
    board_state_np = np.array(board_state, dtype=np.int32)
    for piece in piece_range:
        legal_action_indices = piece_move.get_legal_moves(piece, board_state_np)
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
        moves = piece_move.map_actions_to_legal_moves(np.array([action], dtype=np.int32))
        moves = moves[0].tolist()
        moves = moves[::-1]         # reverse elements

        print(f"AI selected piece {piece} with move {moves}")
        return piece, moves
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

    # Convert to numpy arrays efficiently
    states_np = np.stack(states).astype(np.float32)
    next_states_np = np.stack(next_states).astype(np.float32)
    actions_np = np.array(actions, dtype=np.int64)
    rewards_np = np.array(rewards, dtype=np.float32).reshape(-1, 1)
    dones_np = np.array(dones, dtype=np.float32).reshape(-1, 1)

    # Move to GPU as tensors
    states_tensor = torch.from_numpy(states_np).to(device)
    next_states_tensor = torch.from_numpy(next_states_np).to(device)
    actions_tensor = torch.from_numpy(actions_np).unsqueeze(1).to(device)
    rewards_tensor = torch.from_numpy(rewards_np).to(device)
    dones_tensor = torch.from_numpy(dones_np).to(device)

    # Training mode
    policy_net.train()

    with autocast("cuda"):  # Mixed precision
        policy_output, _ = policy_net(states_tensor)
        current_q_values = policy_output.gather(1, actions_tensor)

        with torch.no_grad():
            target_net.eval()
            next_policy_output, _ = target_net(next_states_tensor)
            next_q_values = next_policy_output.max(1, keepdim=True)[0]
            target_q_values = rewards_tensor + (GAMMA * next_q_values * (1 - dones_tensor))

        loss = F.smooth_l1_loss(current_q_values, target_q_values)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def simulate_game(game_id, red_policy_net, black_policy_net, red_replay_buffer, black_replay_buffer, device):
    global EPSILON
    game.board_1d = game.board_1d_init
    state = game.board_1d
    red_count = 0
    black_count = 0
    turn = 1
    total_red_reward = 0
    total_black_reward = 0
    red_move_history = []
    black_move_history = []

    for t in range(160):
        current_policy_net = red_policy_net if turn == 1 else black_policy_net

        board_1d_np = np.array(state, dtype=np.int32)
        legal_piece_actions = piece_move.generate_all_legal_actions(turn, board_1d_np)

        if not legal_piece_actions:
            break

        if random.random() < EPSILON:
            piece, action = random.choice(legal_piece_actions)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                policy_output, _ = current_policy_net(state_tensor)
                q_values = policy_output.cpu().numpy().squeeze()

            best_q_value = float('-inf')
            best_pair = None
            for p, a in legal_piece_actions:
                if q_values[a] > best_q_value:
                    best_q_value = q_values[a]
                    best_pair = (p, a)
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

        move_history = red_move_history if turn == 1 else black_move_history 
        count = red_count if turn == 1 else black_count
        next_state, reward, done = piece_move.step(piece, action, turn, move_history, count)

        state = convert_if_memoryview(state)
        next_state = convert_if_memoryview(next_state)

        # Store transition
        if turn == 1:
            red_replay_buffer.append((state, action, reward, next_state, done))
            total_red_reward += reward
        else:
            black_replay_buffer.append((state, action, reward, next_state, done))
            total_black_reward += reward

        state = next_state
        turn = 1 - turn

        if done:
            break

    return total_red_reward, total_black_reward


def main():
    global EPSILON

    # initialize networks, optimizers, etc.
    if os.path.exists(red_checkpoint_path) and os.path.exists(black_checkpoint_path) \
    and os.path.exists("red_replay_buffer.pt") and os.path.exists("black_replay_buffer.pt"):
        red_checkpoint = torch.load(red_checkpoint_path)
        black_checkpoint = torch.load(black_checkpoint_path)
        
        red_replay_buffer = torch.load("red_replay_buffer.pt", weights_only=False)
        black_replay_buffer = torch.load("black_replay_buffer.pt", weights_only=False)

        red_policy_net.load_state_dict(red_checkpoint['policy_net'])
        red_target_net.load_state_dict(red_checkpoint['target_net'])
        red_optimizer.load_state_dict(red_checkpoint['optimizer'])

        black_policy_net.load_state_dict(black_checkpoint['policy_net'])
        black_target_net.load_state_dict(black_checkpoint['target_net'])
        black_optimizer.load_state_dict(black_checkpoint['optimizer'])

        batch = max(red_checkpoint['batch'], black_checkpoint['batch'])
        EPSILON = max(red_checkpoint['epsilon'], black_checkpoint['epsilon'])
        print(EPSILON)

        print(f"Starting from batch: {batch}")
    else: 
        red_replay_buffer = deque(maxlen=500000)
        black_replay_buffer = deque(maxlen=500000)
        batch = 0

    EPISODES = 800001
    games_per_batch = 14
    start_time = time.time()

    try:
        for batch_start in range(batch, EPISODES, games_per_batch):
            with ThreadPoolExecutor(max_workers=games_per_batch) as executor:
                futures = [
                    executor.submit(simulate_game, game_id, red_policy_net, black_policy_net, red_replay_buffer, black_replay_buffer, device)
                    for game_id in range(games_per_batch)
                ]
                results = [future.result() for future in futures]

            # After batch of games
            for _ in range(games_per_batch):
                train_dqn(1)   # train red
                train_dqn(-1)  # train black

            """if EPSILON > EPSILON_MIN:
                EPSILON *= EPSILON_DECAY"""

            if batch_start % 100 == 0:
                print(f"Batch {batch_start}, EPSILON: {EPSILON}")

    except KeyboardInterrupt:
        end_time = time.time()
        red_checkpoint = {
                'policy_net': red_policy_net.state_dict(),
                'target_net': red_target_net.state_dict(),
                'optimizer': red_optimizer.state_dict(),
                'batch': batch_start + games_per_batch,
                'epsilon': EPSILON,
            }

        black_checkpoint = {
                'policy_net': black_policy_net.state_dict(),
                'target_net': black_target_net.state_dict(),
                'optimizer': black_optimizer.state_dict(),
                'batch': batch_start + games_per_batch,
                'epsilon': EPSILON,
            }

        torch.save(red_checkpoint, "red_checkpoint.pth")
        torch.save(black_checkpoint, "black_checkpoint.pth")

        torch.save(red_replay_buffer, "red_replay_buffer.pt")
        torch.save(black_replay_buffer, "black_replay_buffer.pt")

        print(f"Checkpoints saved at batch {batch_start + games_per_batch}")
        
        running_time = end_time - start_time
        print("\nRunning time:", running_time, "seconds")


main()

# pip install numpy python-dotenv FastAPi pymysql uvicorn cryptography Cython
# python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
# uvicorn main_api:app --reload