from utils import encode_1d_board_to_board

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
        return 1000
    elif abs_piece in [1, 9]:  # Chariots
        return 700
    elif abs_piece in [10, 11]:  # Cannons
        return 600
    elif abs_piece in [2, 8]:  # Knights
        return 450
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
        if is_square_threatened(new_index, board_1d, turn):
            reward_red -= 100
        
        if 3 <= new_index % 9 <= 5 and 3 <= new_index // 9 <= 6:
            reward_red += 1

        return board_1d, reward_red
    
    elif turn == 0:
        reward_black = pattern_penalty
        old_index = find_piece_1d(piece, board_1d)
        
        if board_1d[new_index] > 0:
            reward_black += get_piece_value(board_1d[new_index])
        
        board_1d[old_index] = 0
        board_1d[new_index] = piece
        if is_square_threatened(new_index, board_1d, turn):
            reward_black -= 100
        
        if 3 <= new_index % 9 <= 5 and 3 <= new_index // 9 <= 6:
            reward_black += 1

        return board_1d, reward_black

def get_legal_moves(piece, board):
    pos = find_piece(piece, board)
    if pos == None:
        return []
    init_x = pos[0]
    init_y = pos[1]

    # Charriot
    if abs(piece) == 1 or abs(piece) == 9:
            legal_moves = []
    
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for dx, dy in directions:
                x, y = init_x, init_y
                
                while True:
                    x += dx
                    y += dy
                    
                    if not (0 <= x < 9 and 0 <= y < 10):
                        break
                    
                    if board[y][x] == 0:
                        legal_moves.append((x, y))
                    elif (board[y][x] < 0 and piece < 0) or (board[y][x] > 0 and piece > 0):  # Friendly piece
                        break
                    elif (board[y][x] < 0 and piece > 0) or (board[y][x] > 0 and piece < 0):  # Enemy piece, valid for capture
                        legal_moves.append((x, y))
                        break
                    
            return legal_moves

    # Horse
    elif abs(piece) == 2 or abs(piece) == 8:
        legal_moves = []

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
                if board[block_y][block_x] == 0:

                    if board[new_y][new_x] == 0:
                        legal_moves.append((new_x, new_y))
                    elif (board[new_y][new_x] > 0 and piece > 0) or (board[new_y][new_x] < 0 and piece < 0):    # friendly piece
                        continue
                    elif (board[new_y][new_x] < 0 and piece > 0) or (board[new_y][new_x] > 0 and piece < 0):  # Enemy piece, valid for capture
                        legal_moves.append((new_x, new_y))
                        continue

        return legal_moves

            
    elif abs(piece) == 3 or abs(piece) == 7:
        legal_moves = []

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

            
    elif abs(piece) == 4 or abs(piece) == 6   :
        legal_moves = []

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
        legal_moves = []

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
        legal_moves = []

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
        legal_moves = []

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

def is_square_threatened(index, board_1d, turn):
    """Check if a square is threatened by opponent pieces."""
    # Convert 1D index to 2D coordinates
    row = index // 9
    col = index % 9
    
    # Create 2D board for easier checking
    board_2d = encode_1d_board_to_board(board_1d)
    
    # Get all opponent's pieces and their legal moves
    opponent_pieces = range(-16, 0) if turn == 1 else range(1, 17)
    
    for piece in opponent_pieces:
        legal_moves = get_legal_moves(piece, board_2d)
        for move_x, move_y in legal_moves:
            if move_x == col and move_y == row:
                return True
    
    return False
