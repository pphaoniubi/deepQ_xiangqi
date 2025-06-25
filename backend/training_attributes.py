import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from game import game
from mcts import *
import concurrent.futures
import multiprocessing as mp
import time

load_dotenv()
game = game()

class XiangqiNet(nn.Module):
    def __init__(self, action_size):
        super(XiangqiNet, self).__init__()

        self.policy_output_size = action_size

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
        policy = self.policy_bn2(self.policy_conv2(policy))  # ← no ReLU here
        policy = policy.view(batch_size, -1)
        policy = F.softmax(self.policy_fc(policy), dim=1)

        # Value Head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(batch_size, -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


def save_training_state(filename, net, replay_buffer, iteration):
    torch.save({
        'model_state_dict': net.state_dict(),
        'replay_buffer': replay_buffer,
        'iteration': iteration
    }, filename)
    print(f"saved at iteration {iteration}")

def load_training_state(filename, net):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['model_state_dict'])
    replay_buffer = checkpoint['replay_buffer']
    iteration = checkpoint['iteration']
    return replay_buffer, iteration + 1

def simulate_one_game(args):
    from piece_move import generate_all_legal_actions_alpha_zero as legal_actions_fn
    from piece_move import apply_action_fn
    from piece_move import is_terminal as is_terminal_fn
    from game import board_init_fn

    net_state_dict, device, simulations, game_idx = args

    print(f"Starting game {game_idx}")
    
    # Rebuild the model inside the subprocess
    net = XiangqiNet(action_size=8100)
    net.load_state_dict(net_state_dict)
    net.eval()
    net.to(device)

    game_data = simulate_game_with_mcts(
        net, device,
        legal_actions_fn, apply_action_fn,
        board_init_fn, is_terminal_fn,
        simulations=simulations
    )

    print(f"Finished game {game_idx}")

    return game_data

def simulate_game_with_mcts(net, device, legal_actions_fn, apply_action_fn, initial_state_fn, is_terminal_fn, simulations=8):
    state = np.array(initial_state_fn(), dtype=np.int32)
    game_data = []
    turn = 1
    move_count = 0
    max_move = 160
    
    while is_terminal_fn(state) == 0  and move_count < max_move:
        root = MCTSNode(state)
        run_mcts(root, net, turn, legal_actions_fn, apply_action_fn, device, simulations)

        # Build MCTS policy π
        visit_counts = {a: child.visit_count for a, child in root.children.items()}
        total_visits = sum(visit_counts.values())
        pi = np.zeros(net.policy_output_size)

        for a, count in visit_counts.items():
            pi[a] = count / total_visits

        # Save training example
        state_tensor = torch.tensor(state).float().to(device)
        game_data.append((state_tensor, pi, turn))

        # Choose action to play
        action = np.random.choice(list(visit_counts.keys()), p=[count / total_visits for count in visit_counts.values()])
        state = apply_action_fn(state, action)
        turn *= -1
        move_count += 1

    # Determine final result
    result = is_terminal_fn(state)

    # Assign final value z
    result = is_terminal_fn(state)
    if move_count > max_move:
        result = 0
    
    training_examples = []
    for s, pi, player in game_data:
        z = result * player
        training_examples.append((s, torch.tensor(pi), torch.tensor(z, dtype=torch.float32)))

    return training_examples


def train_step(net, batch, device, optimizer=None):
    states, pis, zs = zip(*batch)
    states = torch.stack(states).to(device)
    target_pis = torch.stack(pis).to(device)
    target_zs = torch.tensor(zs).float().to(device)

    pred_pis, pred_zs = net(states)
    loss_policy = F.cross_entropy(pred_pis, target_pis)
    loss_value = F.mse_loss(pred_zs.squeeze(), target_zs)
    loss = loss_policy + loss_value

    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def select_move_with_mcts(board_state_1d, turn):
    root = MCTSNode(board_state_1d)

    simulations = 800

    # Manually expand root first
    state_tensor = torch.tensor(root.state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, value = net(state_tensor)
    priors = torch.softmax(policy_logits.squeeze(0).cpu(), dim=0)

    board_np = np.array(root.state, dtype=np.int32).reshape(-1)

    legal_actions = piece_move.generate_all_legal_actions_alpha_zero(turn, board_np)

    priors_masked = torch.zeros_like(priors)
    priors_masked[legal_actions] = priors[legal_actions]

    if len(legal_actions) > 0:
        root.expand(priors_masked)
    root.backpropagate(value.item())  # Optional, if you want root to start with value

    # Now start simulations
    for _ in range(simulations):
        node = root

        # Selection phase
        while node.children:
            node = node.select()

        # Expansion & Evaluation
        state_tensor = torch.tensor(root.state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_logits, value = net(state_tensor)
        priors = torch.softmax(policy_logits.squeeze(0).cpu(), dim=0)

        board_np = np.array(root.state, dtype=np.int32).reshape(-1)
        legal_actions = piece_move.generate_all_legal_actions_alpha_zero(turn, board_np)
        priors_masked = torch.zeros_like(priors)
        priors_masked[legal_actions] = priors[legal_actions]

        if len(legal_actions) > 0:
            node.expand(priors_masked)

        node.backpropagate(value.item())

    # Choose move with highest visit count
    best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
    return best_action


def main_training_loop(net, device, num_iterations=1000, games_per_iteration=50, simulations=800, batch_size=64):
    try:
        global start_iteration
        global end_iteration
        replay_buffer, start_iteration = load_training_state("checkpoint.pth", net)
        end_iteration = start_iteration
        print(f"Resuming from iteration {start_iteration}")

    except FileNotFoundError:
        print("No checkpoint found. Starting fresh.")
        replay_buffer = []
        start_iteration = 0

    for iteration in range(start_iteration, num_iterations):
        start_time = time.time()
        print(f"\n=== Iteration {iteration} ===")

        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            net_state_dict = net.cpu().state_dict()
            args = [
                (net_state_dict, device, simulations, i)
                for i in range(games_per_iteration)
            ]
            results = executor.map(simulate_one_game, args)

            for game_data in results:
                replay_buffer.extend(game_data)

        net.to(device)

        # keep only the most recent N samples
        max_buffer_size = 10000
        if len(replay_buffer) > max_buffer_size:
            replay_buffer = replay_buffer[-max_buffer_size:]

        # Train the network on random batches
        for _ in range(10):  # 10 epochs per iteration
            batch = random.sample(replay_buffer, batch_size)
            train_step(net, batch, device)

        if iteration != 0:
            save_training_state("checkpoint.pth", net, replay_buffer, iteration)
        
        end_iteration = iteration + 1
        end_time = time.time()
        elapsed = end_time - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
    
        print(f"Iteration time: {minutes} min {seconds} sec")


net = XiangqiNet(action_size=8100).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    
    start_time = time.time()

    try:
        mp.set_start_method("spawn", force=True)
        main_training_loop(
            net=net,
            device=device,
            num_iterations=1000,
            games_per_iteration=50,
            simulations=800,
            batch_size=64
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl + C).")
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        num_of_iteration = end_iteration - start_iteration
        print(f"number of iterations: {num_of_iteration}")
        print(f"Elapsed time: {minutes} min {seconds} sec")
        print(f"average time per iteration: {elapsed / num_of_iteration}")
        
# pip install numpy python-dotenv FastAPi pymysql uvicorn cryptography Cython
# python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
# uvicorn main_api:app --reload
# python3.9 setup.py build_ext --inplace