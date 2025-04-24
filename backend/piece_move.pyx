from concurrent.futures import ThreadPoolExecutor
from game_state import game
cimport cython
import numpy as np
cimport numpy as np


def _generate_piece_actions(args):
    piece, board_1d, get_legal_moves_func = args
    legal_moves = get_legal_moves_func(piece, board_1d)
    return [(piece, action) for action in legal_moves]

cpdef list generate_all_legal_actions(int turn, object board_1d, object get_legal_moves_func):
    cdef list args
    cdef list result = []
    cdef int piece
    piece_range = range(1, 17) if turn == 1 else range(-16, 0)
    args = [(piece, board_1d, get_legal_moves_func) for piece in piece_range]

    with ThreadPoolExecutor() as executor:
        for r in executor.map(_generate_piece_actions, args):
            result.extend(r)

    return result
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef str is_winning(int[:] board_1d):
    if find_piece_1d(-5, board_1d) == -1:
        return "Red wins"
    elif find_piece_1d(5, board_1d) == -1:
        return "Black wins"
    else:
        return "Game continues"

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int32_t, ndim=1] map_legal_moves_to_actions(int[:, :] legal_moves):
    cdef Py_ssize_t i, n = legal_moves.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] index = np.empty(n, dtype=np.int32)
    cdef int x, y

    for i in range(n):
        x = legal_moves[i, 0]
        y = legal_moves[i, 1]
        index[i] = y * 9 + x

    return index

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int32_t, ndim=2] map_actions_to_legal_moves(np.ndarray[np.int32_t, ndim=1] actions):
    cdef Py_ssize_t i, n = actions.shape[0]
    cdef np.ndarray[np.int32_t, ndim=2] moves = np.empty((n, 2), dtype=np.int32)
    cdef int action, x, y

    for i in range(n):
        action = actions[i]
        x = action % 9
        y = action // 9
        moves[i, 0] = x
        moves[i, 1] = y

    return moves

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int32_t, ndim=1] get_legal_moves(int piece, int[:] board_1d):
    cdef int[:, :] board = encode_1d_board_to_board(board_1d)
    cdef int init_x = -1, init_y = -1
    cdef int x, y, dx, dy, new_x, new_y, block_x, block_y
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
        return np.empty((0,), dtype=np.int32)

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
        return map_legal_moves_to_actions(np.array(legal_moves, dtype=np.int32).reshape(-1, 2))

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
        return map_legal_moves_to_actions(np.array(legal_moves, dtype=np.int32).reshape(-1, 2))


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

        return map_legal_moves_to_actions(np.array(legal_moves, dtype=np.int32).reshape(-1, 2))


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

        return map_legal_moves_to_actions(np.array(legal_moves, dtype=np.int32).reshape(-1, 2))

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

        return map_legal_moves_to_actions(np.array(legal_moves, dtype=np.int32).reshape(-1, 2))

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
            
        return map_legal_moves_to_actions(np.array(legal_moves, dtype=np.int32).reshape(-1, 2))


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

        return map_legal_moves_to_actions(np.array(legal_moves, dtype=np.int32).reshape(-1, 2))


ctypedef np.int32_t INT32_t


cpdef np.ndarray[np.int32_t, ndim=2] encode_1d_board_to_board(int[:] board_1d):
    cdef np.ndarray[np.int32_t, ndim=2] board_2d = np.empty((10, 9), dtype=np.int32)
    cdef int i, j

    if len(board_1d) != 90:
        raise ValueError("Invalid board size: Expected 90 elements")

    for i in range(10):
        for j in range(9):
            board_2d[i, j] = board_1d[i * 9 + j]

    return board_2d


cpdef np.ndarray[INT32_t, ndim=1] encode_board_to_1d_board(list board):
    cdef int height = len(board)
    cdef int width = len(board[0])  # assumes all rows are equal
    cdef int i, j, index
    cdef np.ndarray[INT32_t, ndim=1] board_flat = np.empty(height * width, dtype=np.int32)

    index = 0
    for i in range(height):
        for j in range(width):
            board_flat[index] = board[i][j]
            index += 1

    return board_flat


cpdef int find_piece_1d(int piece, int[:] board_1d):
    cdef int i
    for i in range(board_1d.shape[0]):
        if board_1d[i] == piece:
            return i
    return -1


cpdef int get_piece_value(int piece):
    cdef int abs_piece = abs(piece)

    if abs_piece == 5:  # General
        return 2000
    elif abs_piece == 1 or abs_piece == 9:  # Chariots
        return 700
    elif abs_piece == 10 or abs_piece == 11:  # Cannons
        return 650
    elif abs_piece == 2 or abs_piece == 8:  # Horses
        return 600
    elif abs_piece == 3 or abs_piece == 7:  # Elephants
        return 350
    elif abs_piece == 4 or abs_piece == 6:  # Advisors
        return 300
    else:  # Pawns
        return 100


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint is_piece_threatened(int index, int[:] board_1d, int turn):
    cdef int target_index = index
    cdef int piece, move_index
    cdef np.ndarray[np.int32_t, ndim=1] legal_moves
    cdef Py_ssize_t i, n

    # Determine opponent pieces
    if turn == 1:
        opponent_pieces = range(-16, 0)
    else:
        opponent_pieces = range(1, 17)

    # Check if any opponent move threatens the target index
    for piece in opponent_pieces:
        legal_moves = get_legal_moves(piece, board_1d)  # must return np.ndarray[int32, ndim=1]
        n = legal_moves.shape[0]
        for i in range(n):
            move_index = legal_moves[i]
            if move_index == target_index:
                return True

    return False


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint is_check(int[:] board_1d, int turn):
    cdef int general
    cdef int general_position_1d
    cdef int piece, move_index
    cdef np.ndarray[np.int32_t, ndim=1] legal_moves
    cdef Py_ssize_t i, n

    # Select the general based on turn
    general = 5 if turn == 1 else -5

    # Find general's position (1D)
    general_position_1d = find_piece_1d(general, board_1d)
    if general_position_1d == -1:
        return False  # General not found

    # Define opponent pieces
    if turn == 1:
        opponent_pieces = range(-16, 0)
    else:
        opponent_pieces = range(1, 17)

    for piece in opponent_pieces:
        legal_moves = get_legal_moves(piece, board_1d)
        n = legal_moves.shape[0]
        for i in range(n):
            move_index = legal_moves[i]
            if move_index == general_position_1d:
                return True

    return False


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint is_check_others(int[:] board_1d, int turn):
    cdef int general
    cdef int general_position_1d
    cdef int piece, move_index
    cdef np.ndarray[np.int32_t, ndim=1] legal_moves
    cdef Py_ssize_t i, n

    if turn == 1:
        ally_pieces = range(1, 17)
        general = -5  # opponent's general
    else:
        ally_pieces = range(-16, 0)
        general = 5

    general_position_1d = find_piece_1d(general, board_1d)
    if general_position_1d == -1:
        return False  # general not found

    for piece in ally_pieces:
        legal_moves = get_legal_moves(piece, board_1d)
        n = legal_moves.shape[0]
        for i in range(n):
            move_index = legal_moves[i]
            if move_index == general_position_1d:
                return True

    return False


cpdef tuple step(int piece, int new_index, int turn, list move_history, int count):
    cdef int[:] board_1d, board_1d_input
    cdef int reward
    cdef bint done
    cdef str winner

    board_1d_input = np.asarray(game.board_1d, dtype=np.int32)
    board_1d, reward = make_move_1d(piece, new_index, board_1d_input, turn, count, move_history)
    game.board_1d = board_1d

    winner = is_winning(board_1d)  # works directly on updated 1D board
    done = (winner == "Red wins" and turn == 1) or (winner == "Black wins" and turn == 0)

    if done and count < 30:
        reward += 3000

    return board_1d, reward, done


cpdef tuple make_move_1d(int piece, int new_index, int[:] board_1d, int turn, int count, list move_history):
    cdef int count_penalty = -80 if count > 30 else 0
    cdef int pattern_penalty = 0
    cdef int piece_move_count = 0
    cdef int old_index = find_piece_1d(piece, board_1d)
    cdef int reward = 0
    cdef int i, move_piece, move_index
    cdef int mv_len = len(move_history)

    # Detect repetition pattern
    if mv_len >= 4:
        move0 = move_history[0]
        move1 = move_history[1]
        move2 = move_history[2]
        move3 = move_history[3]

        if (move0[0] == move2[0] == piece and
            move1[0] == move3[0] and
            move0[1] == move2[1] and
            move1[1] == move3[1]):
            pattern_penalty = -200

    # Count how many times this piece has moved
    for i in range(mv_len):
        move_piece = move_history[i][0]
        if move_piece == piece:
            piece_move_count += 1

    if piece_move_count > 2:
        pattern_penalty -= 10 * (piece_move_count - 2)
    if piece_move_count > 5:
        pattern_penalty -= 30 * (piece_move_count - 2)

    reward += pattern_penalty + count_penalty

    # Reward if capturing
    if turn == 1:  # red
        if board_1d[new_index] < 0:
            reward += get_piece_value(board_1d[new_index])
    else:          # black
        if board_1d[new_index] > 0:
            reward += get_piece_value(board_1d[new_index])

    # Perform the move
    board_1d[old_index] = 0
    board_1d[new_index] = piece

    # Threat and check detection
    if is_piece_threatened(new_index, board_1d, turn):
        reward -= 100
    if is_check(board_1d, turn):
        reward -= 500
    if is_check_others(board_1d, turn):
        reward += 500

    return board_1d, reward  # already a memoryview; no need to wrap