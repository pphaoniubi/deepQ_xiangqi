from training_attributes import *
import random
import numpy as np
import board_piece as board_piece
from game_state import game
import os
import time;


def encode_board_to_1d_board(board):
    board_flat = []
    for row in board:
        for cross in row:
            board_flat.append(cross)
    # print(board)

    return np.array(board_flat)

def encode_1d_board_to_board(board_1d):
    """
    Convert a 1D board representation back to a 2D Xiangqi board.
    
    :param board_1d: List of 90 elements representing the board in 1D format.
    :return: 2D list (10x9) representing the board.
    """
    if len(board_1d) != 90:
        raise ValueError("Invalid board size: Expected 90 elements")

    board_2d = [board_1d[i * 9:(i + 1) * 9] for i in range(10)]  # Convert to 10x9 grid
    return board_2d


def map_legal_moves_to_actions(legal_moves, ACTION_SIZE):
    index = []
    for legal_move in legal_moves:
        index.append(legal_move[1] * 9 + legal_move[0])

    return index

def step(piece, new_index):
    """
    Execute the action in the Pygame game and return the new state, reward, and done status.
    :param action_index: Index of the action to take
    :param action_space_size: Total number of actions
    :param game: Instance of your Pygame Xiangqi game
    :return: New state, reward, done
    """
    reward = 0
    board_1d, reward = board_piece.make_move_1d(piece, new_index, encode_board_to_1d_board(game.board), reward)      # make move on 1D

    game.board = encode_1d_board_to_board(board_1d)

    winner = board_piece.is_winning(game.board)
    if winner == "Red wins":
        reward += 2000
        done = True
    elif winner == "Black wins":
        reward -= 2000
        done = True
    elif winner == "Game continues":
        done = False

    return encode_board_to_1d_board(game.board), reward, done


# load parameters
if os.path.exists(r"C:\Users\Peter\Desktop\deepQ_xiangqi\checkpoint.pth"):

    checkpoint = torch.load("checkpoint.pth")

    print("Saved keys in .pth file:", checkpoint.keys())

    policy_net.load_state_dict(checkpoint['policy_net'])
    target_net.load_state_dict(checkpoint['target_net'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    start_episode = checkpoint['episode']

EPISODES = 100000
# Training loop with Pygame
move_count = 0
start_time = time.time()

try:
    #for episode in range(start_episode, EPISODES):
    for episode in range(0, EPISODES):

        game.board = game.board_init
        state = encode_board_to_1d_board(game.board)  # Reset the Pygame board
        total_reward = 0
        turn = 1
        
        count = 0
        for t in range(200):

            legal_piece_actions = []  # Store all valid (piece, action) pairs
            for piece in range(1, 17) if turn == 1 else range(-16, 0):  # Iterate over all pieces
                legal_moves = board_piece.get_legal_moves(piece, game.board)
                legal_action_indices = map_legal_moves_to_actions(legal_moves, ACTION_SIZE) 

                for action in legal_action_indices:
                    legal_piece_actions.append((piece, action))  # Store valid (piece, action) pairs

            # Choose an action (random or based on policy)
            if random.random() < EPSILON:
                piece, action = random.choice(legal_piece_actions)  # Explore

            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor).cpu().detach().numpy().squeeze()

                # Find the best (piece, action) pair using Q-values
                best_q_value = -float('inf')
                best_pair = None
                for piece, action in legal_piece_actions:
                    if q_values[action] > best_q_value:
                        best_q_value = q_values[action]
                        best_pair = (piece, action)

                piece, action = best_pair  # Select the best piece-action pair

            # Take the action and observe the new state
            next_state, reward, done = step(piece, action) # 这里是1D board
            replay_buffer.append((state, action, reward, next_state, done))

            # Train the network
            train_dqn()                         

            state = next_state
            total_reward += reward

            turn = 1 - turn

            count = t

            if done:
                break

        # Decay epsilon
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        # Update target network periodically
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
            checkpoint = {
                'policy_net': policy_net.state_dict(),
                'target_net': target_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'episode': episode
            }
            
            torch.save(checkpoint, "checkpoint.pth")
            print(f"Checkpoint saved at episode {episode}")


        print(f"Episode {episode}, Total Reward: {total_reward}, Move count: {count}")

except KeyboardInterrupt:
    end_time = time.time()  # Stop timer on Ctrl+C
    running_time = end_time - start_time
    print("\nRunning time:", running_time, "seconds")