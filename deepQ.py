from training_attributes import *
import random
import numpy as np
from attributes import *
from board_piece import *
from game_state import game

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

def step(piece_name, new_x, new_y, init_x, init_y, move_count):
    """
    Execute the action in the Pygame game and return the new state, reward, and done status.
    :param action_index: Index of the action to take
    :param action_space_size: Total number of actions
    :param game: Instance of your Pygame Xiangqi game
    :return: New state, reward, done
    """
    # Decode the action index back to a move
    start = (init_x,  init_y)
    end = (new_x, new_y)
    move = (start, end)

    is_legal = is_move_valid(piece_name, new_x, new_y, init_x, init_y)
    if not is_legal:
        return encode_pieces_to_1d_board(pieces), -10, True  # Invalid move penalty

    make_move(piece_name, new_x, new_y)

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


    return encode_pieces_to_1d_board(pieces), reward, done



EPISODES = 1000
# Training loop with Pygame
move_count = 0
for episode in range(EPISODES):
    state = encode_pieces_to_1d_board(init_board())  # Reset the Pygame board
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
        next_state, reward, done = step(game.piece_name, game.new_x, game.new_y, game.init_x, game.init_y)
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
