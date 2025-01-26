from training_attributes import *
import random
import numpy as np
from attributes import *
from board_piece import *

def encode_pieces_to_1d_board(pieces):
    board = [[0]*9 for i in range(10)]
    for name, (image, rect) in pieces.items():
        i = int((rect.x - 55) / gap)
        j = int((rect.y - 55) / gap)
        if name.find("Black") != -1:
            board[j][i] = -1
        elif name.find("Red") != -1:
            board[j][i] = 1
    board_flat = []
    for row in board:
        for cross in row:
            board_flat.append(cross)
    print(board_flat)

    return np.array(board_flat)

def step(action_index, action_space_size, game):
    """
    Execute the action in the Pygame game and return the new state, reward, and done status.
    :param action_index: Index of the action to take
    :param action_space_size: Total number of actions
    :param game: Instance of your Pygame Xiangqi game
    :return: New state, reward, done
    """
    # Decode the action index back to a move
    start = (action_index // 9 % 10, action_index % 9)
    end = (action_index // 90 % 10, action_index % 90)
    move = (start, end)

    # Apply the move in the Pygame game
    legal_moves = game.get_legal_moves()
    if move not in legal_moves:
        return encode_pieces_to_1d_board(pieces), -10, True  # Invalid move penalty

    game.make_move(move)

    # Determine reward
    reward = 0
    if is_winning() == "Red wins":
        reward = 100
        done = True
    elif is_winning() == "Black wins":
        reward = -100
        done = True
    else:
        done = game.is_draw()

    return encode_pieces_to_1d_board(game.board), reward, done

def map_legal_moves_to_actions(legal_moves, action_space_size):
    """
    Maps legal moves to indices in the fixed action space.
    :param legal_moves: List of legal moves (start_pos, end_pos)
    :param action_space_size: Total number of actions
    :return: List of action indices
    """
    action_indices = []
    for move in legal_moves:
        # Map (start_pos, end_pos) to a unique index
        start, end = move
        action_index = start[0] * 9 + start[1] + end[0] * 90 + end[1]
        if action_index < action_space_size:
            action_indices.append(action_index)
    return action_indices


EPISODES = 1000
# Training loop with Pygame
for episode in range(EPISODES):
    state = encode_board(game.reset())  # Reset the Pygame board
    total_reward = 0

    for t in range(200):  # Max steps per episode
        # Get legal moves and map to action space
        legal_moves = game.get_legal_moves()
        legal_action_indices = map_legal_moves_to_actions(legal_moves, ACTION_SIZE)

        # Choose an action (random or based on policy)
        if random.random() < EPSILON:
            action = random.choice(legal_action_indices)  # Explore
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor).cpu().detach().numpy().squeeze()
            action = legal_action_indices[np.argmax(q_values[legal_action_indices])]

        # Take the action and observe the new state
        next_state, reward, done = step(action, ACTION_SIZE, game)
        replay_buffer.append((state, action, reward, next_state, done))

        # Train the network
        train_dqn()

        state = next_state
        total_reward += reward

        if done:
            break

    # Decay epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    # Update target network periodically
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}")
