import numpy as np
import torch
import torch.nn.functional as F
import piece_move

class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state              # e.g., a NumPy array representing the board
        self.parent = parent            # parent MCTSNode
        self.children = {}             # action -> child MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior             # prior probability from the policy network

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

def backpropagate(path, value):
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value  # Switch perspective for alternating turns


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

        probs = piece_move.masked_softmax(logits, np.array(legal_actions, dtype=np.int32))


        # Store children
        for a in legal_actions:
            next_state = apply_action_fn(node.state, a)
            node.children[a] = MCTSNode(next_state, parent=node, prior=probs[a])

        backpropagate(path, value.item())


def run_mcts(root, net, turn, legal_actions_fn, apply_action_fn, device, simulations=100):
    leaf_nodes = []

    if not root.children:
        expand_node(root, turn, legal_actions_fn, apply_action_fn)

    for _ in range(simulations):
        node = root
        path = [node]

        while node.children:
            # Build flat dicts for Cython input
            values = {a: child.value() for a, child in node.children.items()}
            priors = {a: child.prior for a, child in node.children.items()}
            visit_counts = {a: child.visit_count for a, child in node.children.items()}

            # Call Cython-accelerated selection
            action = piece_move.select_child(
                node.children,
                node.visit_count,
                values,
                priors,
                visit_counts,
                c_puct=1.0  # you can make this tunable
            )

            # Get selected child
            node = node.children[action]
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