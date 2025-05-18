import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from game_state import game
import piece_move
from mcts import MCTSNode

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
    
def simulate_game_with_mcts(net, device, legal_actions_fn, apply_action_fn, initial_state_fn, is_terminal_fn, get_winner_fn, simulations=800):
    state = initial_state_fn()
    game_data = []
    current_player = 1  # 1 for Red, -1 for Black

    while not is_terminal_fn(state):
        root = MCTSNode(state)
        MCTSNode.run_mcts(root, net, legal_actions_fn, apply_action_fn, device, simulations)

        # Build MCTS policy π
        visit_counts = {a: child.visit_count for a, child in root.children.items()}
        total_visits = sum(visit_counts.values())
        pi = np.zeros(net.policy_output_size)
        for a, count in visit_counts.items():
            pi[a] = count / total_visits

        # Save training example
        state_tensor = torch.tensor(state).float().to(device)
        game_data.append((state_tensor, pi, current_player))

        # Choose action to play
        action = np.random.choice(list(visit_counts.keys()), p=[count / total_visits for count in visit_counts.values()])
        state = apply_action_fn(state, action)
        current_player *= -1

    # Assign final value z
    result = get_winner_fn(state)  # 1 (Red win), -1 (Black win), or 0 (draw)
    training_examples = []
    for s, pi, player in game_data:
        z = result * player
        training_examples.append((s, torch.tensor(pi), torch.tensor(z, dtype=torch.float32)))

    return training_examples



# pip install numpy python-dotenv FastAPi pymysql uvicorn cryptography Cython
# python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
# uvicorn main_api:app --reload