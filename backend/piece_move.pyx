from game_state import game
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt

np.import_array()

cpdef int move_to_index(int from_idx, int to_idx):
    return from_idx * 90 + to_idx

cpdef tuple index_to_move(int index):
    cdef int from_idx = index // 90
    cdef int to_idx = index % 90
    return (from_idx, to_idx)

cpdef np.ndarray[np.int32_t, ndim=1] apply_action_fn(int[:] board_1d, int action_index):
    cdef int from_idx = action_index // 90
    cdef int to_idx = action_index % 90
    cdef np.ndarray[np.int32_t, ndim=1] new_board = np.empty(90, dtype=np.int32)
    cdef int i

    # Copy board content manually for speed and memoryview compatibility
    for i in range(90):
        new_board[i] = board_1d[i]

    new_board[to_idx] = board_1d[from_idx]
    new_board[from_idx] = 0

    return new_board
    
cpdef list generate_all_legal_actions_alpha_zero(int turn, object board_1d_obj):
    cdef np.ndarray[np.int32_t, ndim=1] board_np
    cdef int[:] board_1d
    cdef list result = []
    cdef int piece, index, from_pos, to_pos
    cdef np.ndarray[np.int32_t, ndim=1] to_pos_arr

    # Validate and convert input
    if not isinstance(board_1d_obj, np.ndarray):
        raise TypeError("Expected a NumPy array")

    board_np = np.ascontiguousarray(board_1d_obj, dtype=np.int32)

    if board_np.ndim != 1 or board_np.shape[0] != 90:
        raise ValueError("Expected a 1D array of length 90")

    board_1d = board_np  # this is now safe

    for index in range(90):
        piece = board_1d[index]
        if piece == 0:
            continue
        
        elif turn == 1 and piece > 0:
            from_pos = index
            to_pos_arr = get_legal_moves(piece, board_1d)
            for i in range(to_pos_arr.shape[0]):
                result.append(move_to_index(from_pos, to_pos_arr[i]))

        elif turn == -1 and piece < 0:
            from_pos = index
            to_pos_arr = get_legal_moves(piece, board_1d)
            for i in range(to_pos_arr.shape[0]):
                result.append(move_to_index(from_pos, to_pos_arr[i]))

    return result
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int is_terminal(int[:] board_1d):
    if find_piece_1d(-5, board_1d) == -1:
        return 1
    elif find_piece_1d(5, board_1d) == -1:
        return -1
    else:
        return 0

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
                if (piece < 0 and new_y <= 4) or (piece > 0 and new_y >= 5):
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



cdef double EPSILON = 1e-8

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ucb_score(int parent_visits, double child_value, double child_prior, int child_visits, double c_puct=1.0):
    cdef double u_value
    u_value = c_puct * child_prior * sqrt(parent_visits) / (1.0 + child_visits)
    return child_value + u_value

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int select_child(dict children, int parent_visits, dict values, dict priors, dict visit_counts, double c_puct=1.0):
    cdef double best_score = -1e9
    cdef int best_action = -1
    cdef double score
    cdef int a
    for a in children:
        score = ucb_score(
            parent_visits,
            values[a],
            priors[a],
            visit_counts[a],
            c_puct
        )
        if score > best_score:
            best_score = score
            best_action = a
    return best_action

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray masked_softmax(np.ndarray logits, np.ndarray legal_actions):
    cdef int i
    cdef int action
    cdef int size = logits.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1] exp_logits = np.zeros(size, dtype=np.float32)
    cdef float max_logit = np.max(logits)
    cdef float sum_exp = 0.0

    for i in range(legal_actions.shape[0]):
        action = legal_actions[i]
        exp_logits[action] = exp(logits[action] - max_logit)
        sum_exp += exp_logits[action]

    if sum_exp < EPSILON:
        # Fallback to uniform
        for i in range(legal_actions.shape[0]):
            exp_logits[legal_actions[i]] = 1.0 / legal_actions.shape[0]
    else:
        for i in range(legal_actions.shape[0]):
            action = legal_actions[i]
            exp_logits[action] /= sum_exp

    return exp_logits


cpdef tuple make_move_1d(int piece, int new_index, int[:] board_1d):
    cdef int old_index = find_piece_1d(piece, board_1d)

    board_1d[old_index] = 0
    board_1d[new_index] = piece

    return board_1d