import numpy as np
import torch
import torch.nn.functional as F

class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    def value(self):
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count

def ucb_score(parent, child, c_puct=1.0):
    prior = child.prior
    q_value = child.value()
    u_value = c_puct * prior * np.sqrt(parent.visit_count) / (1 + child.visit_count)
    return q_value + u_value

def select_child(node):
    return max(node.children.items(), key=lambda item: ucb_score(node, item[1]))

def expand_and_evaluate(node, net, turn, legal_actions_fn, apply_action_fn, device):
    state_tensor = torch.tensor(node.state).float().unsqueeze(0).to(device)
    policy, value = net(state_tensor)
    policy = policy.squeeze(0).detach().cpu().numpy()
    value = value.item()

    legal_actions = legal_actions_fn(node.state, turn)
    policy = policy * np.isin(np.arange(len(policy)), legal_actions)  # mask illegal
    policy /= np.sum(policy) + 1e-8

    for a in legal_actions:
        next_state = apply_action_fn(node.state, a)
        node.children[a] = MCTSNode(next_state, parent=node, prior=policy[a])

    return value

def backpropagate(path, value):
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value  # flip for opponent

def run_mcts(root, net, turn, legal_actions_fn, apply_action_fn, device, simulations=100):
    for _ in range(simulations):
        node = root
        path = [node]

        while node.children:
            action, node = select_child(node)
            path.append(node)

        value = expand_and_evaluate(node, net, turn, legal_actions_fn, apply_action_fn, device)
        backpropagate(path, value)

    return root
