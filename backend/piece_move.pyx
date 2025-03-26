# piece_move.pyx

# Optional Cython optimizations
from libc.stdlib cimport malloc, free
cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_all_legal_actions(int turn, int[:, :] board, object get_legal_moves_func, object map_func):
    cdef int i, piece
    cdef list result = []
    piece_range = list(range(1, 17)) if turn == 1 else list(range(-16, 0))
    cdef list legal_moves
    cdef list legal_action_indices
    cdef int action

    for i in range(len(piece_range)):
        piece = piece_range[i]
        legal_moves = get_legal_moves_func(piece, board)
        legal_action_indices = map_func(legal_moves)

        for action in legal_action_indices:
            result.append((piece, action))

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def map_legal_moves_to_actions(list legal_moves):
    cdef list index = []
    cdef int x, y
    for x, y in legal_moves:
        index.append(y * 9 + x)
    return index

@cython.boundscheck(False)
@cython.wraparound(False)
def get_legal_moves(int piece, int[:, :] board):
    cdef int init_x = -1, init_y = -1
    cdef int y, x
    cdef list legal_moves = []

    # Find the piece on the board
    for y in range(10):
        for x in range(9):
            if board[y, x] == piece:
                init_x, init_y = x, y
                break
        if init_x != -1:
            break

    if init_x == -1:
        return []

    # Chariot
    if abs(piece) == 1 or abs(piece) == 9:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            x, y = init_x, init_y
            while True:
                x += dx
                y += dy
                if not (0 <= x < 9 and 0 <= y < 10):
                    break
                if board[y, x] == 0:
                    legal_moves.append((x, y))
                elif (board[y, x] < 0 and piece < 0) or (board[y, x] > 0 and piece > 0):
                    break
                elif (board[y, x] < 0 and piece > 0) or (board[y, x] > 0 and piece < 0):
                    legal_moves.append((x, y))
                    break
        return legal_moves

    # Horse
    elif abs(piece) == 2 or abs(piece) == 8:
        horse_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (-1, 2), (1, -2), (-1, -2)
        ]
        block_positions = {
            (2, 1): (init_x + 1, init_y),
            (2, -1): (init_x + 1, init_y),
            (-2, 1): (init_x - 1, init_y),
            (-2, -1): (init_x - 1, init_y),
            (1, 2): (init_x, init_y + 1),
            (-1, 2): (init_x, init_y + 1),
            (1, -2): (init_x, init_y - 1),
            (-1, -2): (init_x, init_y - 1),
        }
        for dx, dy in horse_moves:
            new_x, new_y = init_x + dx, init_y + dy
            if 0 <= new_x < 9 and 0 <= new_y < 10:
                block_x, block_y = block_positions[(dx, dy)]
                if board[block_y, block_x] == 0:
                    if board[new_y, new_x] == 0:
                        legal_moves.append((new_x, new_y))
                    elif (board[new_y, new_x] > 0 and piece > 0) or (board[new_y, new_x] < 0 and piece < 0):
                        continue
                    elif (board[new_y, new_x] < 0 and piece > 0) or (board[new_y, new_x] > 0 and piece < 0):
                        legal_moves.append((new_x, new_y))
        return legal_moves


    elif abs(piece) == 3 or abs(piece) == 7:
        elephant_moves = [
            (2, 2), (2, -2), (-2, 2), (-2, -2)
        ]

        block_positions = {
            (2, 2): (init_x + 1, init_y + 1),
            (2, -2): (init_x + 1, init_y - 1),
            (-2, 2): (init_x - 1, init_y + 1),
            (-2, -2): (init_x - 1, init_y - 1),
        }

        for dx, dy in elephant_moves:
            new_x, new_y = init_x + dx, init_y + dy

            if 0 <= new_x < 9 and 0 <= new_y < 10:
                if (new_y <= 4) or (new_y >= 5):
                    block_x, block_y = block_positions[(dx, dy)]
                    if board[block_y][block_x] == 0:

                        if board[new_y][new_x] == 0:
                            legal_moves.append((new_x, new_y))
                        elif (board[new_y][new_x] > 0 and piece > 0) or (board[new_y][new_x] < 0 and piece < 0):    # friendly piece
                            continue
                        elif (board[new_y][new_x] < 0 and piece > 0) or (board[new_y][new_x] > 0 and piece < 0):  # Enemy piece, valid for capture
                            legal_moves.append((new_x, new_y))

        return legal_moves


    elif abs(piece) == 4 or abs(piece) == 6:
        advisor_moves = [
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        for dx, dy in advisor_moves:
            new_x, new_y = init_x + dx, init_y + dy

            if (3 <= new_x <= 5 and 0 <= new_y <= 2) or (3 <= new_x <= 5 and 7 <= new_y <= 9):

                if board[new_y][new_x] == 0:
                    legal_moves.append((new_x, new_y))
                elif (board[new_y][new_x] > 0 and piece > 0) or (board[new_y][new_x] < 0 and piece < 0):    # friendly piece
                    continue
                elif (board[new_y][new_x] < 0 and piece > 0) or (board[new_y][new_x] > 0 and piece < 0):  # Enemy piece, valid for capture
                    legal_moves.append((new_x, new_y))

        return legal_moves

    elif abs(piece) == 5:
        general_moves = [
            (1, 0), (0, 1), (-1, 0), (0, -1)
        ]

        for dx, dy in general_moves:
            new_x, new_y = init_x + dx, init_y + dy

            if (3 <= new_x <= 5 and 0 <= new_y <= 2) or (3 <= new_x <= 5 and 7 <= new_y <= 9):
                if board[new_y][new_x] == 0:
                    legal_moves.append((new_x, new_y))
                elif (board[new_y][new_x] > 0 and piece > 0) or (board[new_y][new_x] < 0 and piece < 0):    # friendly piece
                    continue
                elif (board[new_y][new_x] < 0 and piece > 0) or (board[new_y][new_x] > 0 and piece < 0):  # Enemy piece, valid for capture
                    legal_moves.append((new_x, new_y))

        return legal_moves

    elif abs(piece) == 10 or abs(piece) == 11:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dx, dy in directions:
            x, y = init_x, init_y
            passed_piece = 0

            while True:
                x += dx
                y += dy

                if not (0 <= x < 9 and 0 <= y < 10):
                    break
                
                if board[y][x] == 0 and passed_piece == 0:
                    legal_moves.append((x, y))
                elif board[y][x] != 0:
                    if passed_piece == 0:
                        passed_piece += 1
                    else: 
                        if (board[y][x] > 0 and piece < 0) or (board[y][x] < 0 and piece > 0):
                            legal_moves.append((x, y))
                            break
                        else: 
                            break
            
        return legal_moves


    elif abs(piece) in (12, 13, 14, 15, 16): 
        if piece > 0:
            has_crossed_river = (init_y <= 4)
            moves = [(0, -1)] 
            if has_crossed_river:
                moves += [(-1, 0), (1, 0)] 
        
        else: 
            has_crossed_river = (init_y >= 5)
            moves = [(0, 1)]
            if has_crossed_river:
                moves += [(-1, 0), (1, 0)] 

        for dx, dy in moves:
            new_x, new_y = init_x + dx, init_y + dy

            if not (0 <= new_x < 9 and 0 <= new_y < 10):
                continue
            elif board[new_y][new_x] == 0:
                legal_moves.append((new_x, new_y))
            elif (board[new_y][new_x] > 0 and piece > 0) or (board[new_y][new_x] < 0 and piece < 0):    # friendly piece
                continue
            elif (board[new_y][new_x] < 0 and piece > 0) or (board[new_y][new_x] > 0 and piece < 0):  # Enemy piece, valid for capture
                legal_moves.append((new_x, new_y))

        return legal_moves