from training_attributes import *
import random
import numpy as np
from attributes import *
import board_piece
from game_state import game

piece_encoding = {
    "Red Chariot": 1, "Black Chariot": -1,
    "Red Horse": 2, "Black Horse": -2,
    "Red Elephant": 3, "Black Elephant": -3,
    "Red Advisor": 4, "Black Advisor": -4,
    "Red General": 5, "Black General": -5,
    "Red Cannon": 6, "Black Cannon": -6,
    "Red Soldier": 7, "Black Soldier": -7
}



def encode_board_to_1d_board(board):
    board_flat = []
    for row in board:
        for cross in row:
            board_flat.append(cross)
    print(board_flat)

    return np.array(board_flat)

def map_legal_moves_to_actions(legal_moves, ACTION_SIZE):
    index = []
    for legal_move in legal_moves:
        index.append(legal_move[1] * 9 + legal_move[0])

    return index

def step(piece, new_x, new_y, init_x, init_y, move_count):
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

    legal_moves = board_piece.get_legal_moves(piece, game.board)
    if not legal_moves:
        return encode_board_to_1d_board(game.board), -10, True  # Invalid move penalty

    board_piece.make_move1(piece, new_x, new_y, game.board)

    # Determine reward
    reward = 0
    winner = board_piece.is_winning()
    if winner == "Red wins":
        reward = 100
        done = True
    elif winner == "Black wins":
        reward = -100
        done = True
    elif winner == "Game continues":
        done = False
    else:
        done = game.is_draw()

    return encode_board_to_1d_board(game.board), reward, done


EPISODES = 1000
# Training loop with Pygame
move_count = 0
for episode in range(EPISODES):
    state = encode_board_to_1d_board(game.board)  # Reset the Pygame board
    total_reward = 0

    for t in range(200):  # Max steps per episode
        # Get legal moves and map to action space
        legal_moves = game.get_legal_moves(game.piece_name)
        legal_action_indices = map_legal_moves_to_actions(legal_moves, ACTION_SIZE)        # 1d space

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
