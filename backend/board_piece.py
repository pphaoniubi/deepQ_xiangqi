from utils import encode_1d_board_to_board
import piece_move
import numpy as np

def change_sides(side):
    if side == "Red":
        side = side.replace("Red", "Black")
    elif side == "Black":
        side = side.replace("Black", "Red")
    
    return side


def is_winning(board):
    if find_piece(-5, board) == None:
        return "Red wins"
    elif find_piece(5, board) == None:
        return "Black wins"
    else:
        return "Game continues"
    

def find_piece(piece, board):
    found = False
    for j in range(len(board)):
        for i in range(len(board[j])):
            if board[j][i] == piece:
                found = True
                init_x = i
                init_y = j
                break
        if found:
            break
    if found:
        return (init_x, init_y)
    else:
        return None

def find_piece_1d(piece, board_1d):
    for i in range(len(board_1d)):
        if board_1d[i] == piece:
            return i
    else:
        return None


def make_move(piece, new_x, new_y, board):
    init_x, init_y = find_piece(piece, board)
    board[init_y][init_x] = 0
    board[new_y][new_x] = piece

    return board


def get_piece_value(piece):
    """Return the relative value of each piece type."""
    abs_piece = abs(piece)
    if abs_piece == 5:  # General
        return 2000
    elif abs_piece in [1, 9]:  # Chariots
        return 700
    elif abs_piece in [10, 11]:  # Cannons
        return 650
    elif abs_piece in [2, 8]:  # Horses
        return 600
    elif abs_piece in [3, 7]:  # Elephants
        return 350
    elif abs_piece in [4, 6]:  # Advisors
        return 300
    else:  # Pawns
        return 100

def make_move_1d(piece, new_index, board_1d, turn, move_history):
    pattern_penalty = 0
    
    if move_history and len(move_history) >= 2:
        if (len(move_history) >= 4 and
              move_history[0][0] == move_history[2][0] == piece and
              move_history[1][0] == move_history[3][0] and
              move_history[0][1] == move_history[2][1] and
              move_history[1][1] == move_history[3][1]):
            pattern_penalty = -200
            # print(f"A->B->A->B pattern penalty applied on piece {piece}")
        
    # Add progressive penalty based on how many times this piece has moved
    piece_move_count = sum(1 for move in move_history if move[0] == piece)
    if piece_move_count > 2:
        pattern_penalty -= 10 * (piece_move_count - 2)  # Progressive penalty for moving same piece too much

    if piece_move_count > 5:
        pattern_penalty -= 30 * (piece_move_count - 2)  # Progressive penalty for moving same piece too much
    
    if turn == 1:
        reward_red = pattern_penalty
        old_index = find_piece_1d(piece, board_1d)
        
        if board_1d[new_index] < 0:
            reward_red += get_piece_value(board_1d[new_index])
        
        board_1d[old_index] = 0
        board_1d[new_index] = piece
        if is_piece_threatened(new_index, board_1d, turn):
            reward_red -= 100
        
        if is_check(board_1d, turn):
            reward_red -= 200

        if is_check_others(board_1d, turn):
            reward_red += 500

        return board_1d, reward_red
    
    elif turn == 0:
        reward_black = pattern_penalty
        old_index = find_piece_1d(piece, board_1d)
        
        if board_1d[new_index] > 0:
            reward_black += get_piece_value(board_1d[new_index])
        
        board_1d[old_index] = 0
        board_1d[new_index] = piece
        if is_piece_threatened(new_index, board_1d, turn):
            reward_black -= 100

        if is_check(board_1d, turn):
            reward_black -= 200

        if is_check_others(board_1d, turn):
            reward_black += 500

        return board_1d, reward_black

def is_piece_threatened(index, board_1d, turn):
    row = index // 9
    col = index % 9
    
    # Create 2D board for easier checking
    board_2d = encode_1d_board_to_board(board_1d)
    
    # Get all opponent's pieces and their legal moves
    opponent_pieces = range(-16, 0) if turn == 1 else range(1, 17)
    board_np = np.array(board_2d, dtype=np.int32)

    for piece in opponent_pieces:
        legal_moves = piece_move.get_legal_moves(piece, board_np)
        for move_x, move_y in legal_moves:
            if move_x == col and move_y == row:
                return True
    
    return False

def is_check(board_1d, turn):
    general = 5 if turn == 1 else -5
    board_2d = encode_1d_board_to_board(board_1d)
    general_position = find_piece(general, board_2d)

    if general_position:
        row, col = general_position
    else:
        return False

    opponent_pieces = range(-16, 0) if turn == 1 else range(1, 17)
    board_np = np.array(board_2d, dtype=np.int32)

    for piece in opponent_pieces:
        legal_moves = piece_move.get_legal_moves(piece, board_np)
        for move_x, move_y in legal_moves:
            if move_x == row and move_y == col:
                return True
    
    return False

def is_check_others(board_1d, turn):
    general = -5 if turn == 1 else 5
    board_2d = encode_1d_board_to_board(board_1d)
    general_position = find_piece(general, board_2d)

    if general_position:
        row, col = general_position
    else:
        return False

    opponent_pieces = range(1, 17) if turn == 1 else range(-16, 0)
    board_np = np.array(board_2d, dtype=np.int32)

    for piece in opponent_pieces:
        legal_moves = piece_move.get_legal_moves(piece, board_np)
        for move_x, move_y in legal_moves:
            if move_x == row and move_y == col:
                return True
    
    return False