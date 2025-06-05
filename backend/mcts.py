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

def expand_batch(leaf_nodes, net, turn, legal_actions_fn, apply_action_fn, device):
    # 1. Prepare batched state tensor
    states = []
    for node, _ in leaf_nodes:
        if torch.is_tensor(node.state):
            states.append(node.state.unsqueeze(0))
        else:
            states.append(torch.tensor(node.state, dtype=torch.float32).unsqueeze(0))
    state_batch = torch.cat(states).to(device)

    # 1. Get logits (not softmaxed yet)
    with torch.no_grad():
        policy_logits, values = net(state_batch)  # logits: (B, 8100)

    # 2. Convert to NumPy safely
    policy_logits = policy_logits.cpu().numpy()
    values = values.cpu().numpy()

    # 3. Apply softmax + masking per sample
    for (node, path), logits, value in zip(leaf_nodes, policy_logits, values):
        legal_actions = legal_actions_fn(turn, node.state)

        # Softmax over logits
        max_logit = np.max(logits)  # for numerical stability
        exp_logits = np.exp(logits - max_logit)
        mask = np.zeros_like(exp_logits)
        mask[legal_actions] = 1
        exp_logits *= mask

        sum_exp = np.sum(exp_logits)
        if sum_exp == 0:
            print("Masked softmax is all zero. Falling back to uniform.")
            probs = np.zeros_like(logits)
            probs[legal_actions] = 1 / len(legal_actions)
        else:
            probs = exp_logits / sum_exp

        # Store children
        for a in legal_actions:
            next_state = apply_action_fn(node.state, a)
            node.children[a] = MCTSNode(next_state, parent=node, prior=probs[a])

        backpropagate(path, value.item())


def backpropagate(path, value):
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value  # flip for opponent

def run_mcts(root, net, turn, legal_actions_fn, apply_action_fn, device, simulations=100):
    leaf_nodes = []

    if not root.children:
        expand_node(root, turn, legal_actions_fn, apply_action_fn)

    for _ in range(simulations):
        node = root
        path = [node]

        while node.children:
            action, node = select_child(node)
            path.append(node)

        leaf_nodes.append((node, path))

    # Batch evaluate all leaf nodes
    expand_batch(leaf_nodes, net, turn, legal_actions_fn, apply_action_fn, device)

    return root

def expand_node(node, turn, legal_actions_fn, apply_action_fn):
    legal_actions = legal_actions_fn(turn, node.state)
    for action in legal_actions:
        next_state = apply_action_fn(node.state, action)
        node.children[action] = MCTSNode(state=next_state, parent=node, prior=action)