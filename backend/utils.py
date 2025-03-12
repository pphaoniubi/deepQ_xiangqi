import numpy as np

def encode_board_to_1d_board(board):
    board_flat = []
    for row in board:
        for cross in row:
            board_flat.append(cross)
    return np.array(board_flat)

def encode_1d_board_to_board(board_1d):
    if len(board_1d) != 90:
        raise ValueError("Invalid board size: Expected 90 elements")
    board_2d = [board_1d[i * 9:(i + 1) * 9] for i in range(10)]
    return board_2d 