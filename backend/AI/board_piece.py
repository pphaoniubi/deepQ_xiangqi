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
    
# Sorry for removing your make_move func

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


def make_move_1d(piece, new_index, board_1d, reward):
    old_index = find_piece_1d(piece, board_1d)
    if board_1d[new_index] > 0:
        reward -= 10
    elif board_1d[new_index] < 0:
        reward += 10
    board_1d[old_index] = 0
    board_1d[new_index] = piece

    return board_1d, reward

def get_legal_moves(piece, board):
    pos = find_piece(piece, board)
    if pos == None:
        return []
    init_x = pos[0]
    init_y = pos[1]

    if abs(piece) == 1 or abs(piece) == 9:
            legal_moves = []
    
            # Directions for movement: (delta_x, delta_y)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Gauche, Droite, Haut, Bas
            
            for dx, dy in directions:
                x, y = init_x, init_y
                
                while True:
                    x += dx
                    y += dy
                    
                    # Check if the position is out of bounds
                    if not (0 <= x < 9 and 0 <= y < 10):
                        break  # Stop if out of bounds
                    
                    # Check if the destination is empty or contains an enemy piece
                    if board[y][x] == 0:  # Empty space, valid move
                        legal_moves.append((x, y))
                    elif (board[y][x] < 0 and piece < 0) or (board[y][x] > 0 and piece > 0):  # Friendly piece
                        break
                    elif (board[y][x] < 0 and piece > 0) or (board[y][x] > 0 and piece < 0):  # Enemy piece, valid for capture
                        legal_moves.append((x, y))
                        break  # Stop after capturing
                    
            return legal_moves


    elif abs(piece) == 2 or abs(piece) == 8:
        legal_moves = []

        # Liste des 8 mouvements possibles
        horse_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (-1, 2), (1, -2), (-1, -2)
        ]

        # Vérifier si une pièce bloque la "jambe" (point intermédiaire)
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

            # Vérifier que la destination est sur le plateau
            if 0 <= new_x < 9 and 0 <= new_y < 10:
                # Vérifier si la "jambe" est bloquée
                block_x, block_y = block_positions[(dx, dy)]
                if board[block_y][block_x] == 0:  # Pas d'obstacle
                    # Vérifier que la case de destination est vide ou contient un ennemi
                    if board[new_y][new_x] == 0:
                        legal_moves.append((new_x, new_y))
                    elif (board[new_y][new_x] > 0 and piece > 0) or (board[new_y][new_x] < 0 and piece < 0):    # friendly piece
                        break
                    elif (board[new_y][new_x] < 0 and piece > 0) or (board[new_y][new_x] > 0 and piece < 0):  # Enemy piece, valid for capture
                        legal_moves.append((new_x, new_y))
                        break  # Stop after capturing

        return legal_moves

            
    elif abs(piece) == 3 or abs(piece) == 7:
        legal_moves = []

        # Liste des 8 mouvements possibles
        elephant_moves = [
            (2, 2), (2, -2), (-2, 2), (-2, -2)
        ]

        # Vérifier si une pièce bloque la "jambe" (point intermédiaire)
        block_positions = {
            (2, 2): (init_x + 1, init_y + 1),
            (2, -2): (init_x + 1, init_y - 1),
            (-2, 2): (init_x - 1, init_y + 1),
            (-2, -2): (init_x - 1, init_y - 1),
        }

        for dx, dy in elephant_moves:
            new_x, new_y = init_x + dx, init_y + dy

            # Vérifier que la destination est sur le plateau
            if 0 <= new_x < 9 and 0 <= new_y < 10:
                if (new_y <= 4) or (new_y >= 5):
                    # Vérifier si la "jambe" est bloquée
                    block_x, block_y = block_positions[(dx, dy)]
                    if board[block_y][block_x] == 0:  # Pas d'obstacle
                        # Vérifier que la case de destination est vide ou contient un ennemi
                        if board[new_y][new_x] == 0:
                            legal_moves.append((new_x, new_y))
                        elif (board[new_y][new_x] > 0 and piece > 0) or (board[new_y][new_x] < 0 and piece < 0):    # friendly piece
                            continue
                        elif (board[new_y][new_x] < 0 and piece > 0) or (board[new_y][new_x] > 0 and piece < 0):  # Enemy piece, valid for capture
                            legal_moves.append((new_x, new_y))

        return legal_moves

            
    elif abs(piece) == 4 or abs(piece) == 6   :
        legal_moves = []

        # Liste des 8 mouvements possibles
        advisor_moves = [
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        for dx, dy in advisor_moves:
            new_x, new_y = init_x + dx, init_y + dy

            # Vérifier que la destination est sur le plateau
            if (3 <= new_x <= 5 and 0 <= new_y <= 2) or (3 <= new_x <= 5 and 7 <= new_y <= 9):
                # Vérifier que la case de destination est vide ou contient un ennemi
                if board[new_y][new_x] == 0:
                    legal_moves.append((new_x, new_y))
                elif (board[new_y][new_x] > 0 and piece > 0) or (board[new_y][new_x] < 0 and piece < 0):    # friendly piece
                    continue
                elif (board[new_y][new_x] < 0 and piece > 0) or (board[new_y][new_x] > 0 and piece < 0):  # Enemy piece, valid for capture
                    legal_moves.append((new_x, new_y))

        return legal_moves
    

    elif abs(piece) == 5:
        legal_moves = []

        # Liste des 8 mouvements possibles
        general_moves = [
            (1, 0), (0, 1), (-1, 0), (0, -1)
        ]

        for dx, dy in general_moves:
            new_x, new_y = init_x + dx, init_y + dy

            # Vérifier que la destination est sur le plateau
            if (3 <= new_x <= 5 and 0 <= new_y <= 2) or (3 <= new_x <= 5 and 7 <= new_y <= 9):
                # Vérifier que la case de destination est vide ou contient un ennemi
                if board[new_y][new_x] == 0:
                    legal_moves.append((new_x, new_y))
                elif (board[new_y][new_x] > 0 and piece > 0) or (board[new_y][new_x] < 0 and piece < 0):    # friendly piece
                    continue
                elif (board[new_y][new_x] < 0 and piece > 0) or (board[new_y][new_x] > 0 and piece < 0):  # Enemy piece, valid for capture
                    legal_moves.append((new_x, new_y))

        return legal_moves

            
    elif abs(piece) == 10 or abs(piece) == 11:
        legal_moves = []

        # Directions de mouvement (horizontalement et verticalement)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Gauche, Droite, Haut, Bas
        
        for dx, dy in directions:
            x, y = init_x, init_y
            passed_piece = 0  # Indicateur pour savoir si une pièce a été rencontrée

            while True:
                x += dx
                y += dy

                # Vérifier si la position est hors limites
                if not (0 <= x < 9 and 0 <= y < 10):
                    break  # Stop si hors du plateau
                
                # Vérifier les cases
                if board[y][x] == 0 and passed_piece == 0:  # Case vide
                    legal_moves.append((x, y))
                elif board[y][x] != 0:  # Pièce alliée
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

        # Vérifier si le soldat a traversé la rivière
        if piece > 0:  # Soldat rouge (兵) avance vers le bas
            has_crossed_river = (init_y <= 4)
            moves = [(0, -1)]  # Avancer uniquement vers le bas
            if has_crossed_river:
                moves += [(-1, 0), (1, 0)]  # Peut aller à gauche et à droite après la rivière
        
        else:  # Soldat noir (卒) avance vers le haut
            has_crossed_river = (init_y >= 5)
            moves = [(0, 1)]  # Avancer uniquement vers le haut
            if has_crossed_river:
                moves += [(-1, 0), (1, 0)]  # Peut aller à gauche et à droite après la rivière

        # Vérifier les mouvements possibles
        for dx, dy in moves:
            new_x, new_y = init_x + dx, init_y + dy

            # Vérifier si la position est valide
            if not (0 <= new_x < 9 and 0 <= new_y < 10):
                break
            elif board[new_y][new_x] == 0:
                legal_moves.append((new_x, new_y))
            elif (board[new_y][new_x] > 0 and piece > 0) or (board[new_y][new_x] < 0 and piece < 0):    # friendly piece
                continue
            elif (board[new_y][new_x] < 0 and piece > 0) or (board[new_y][new_x] > 0 and piece < 0):  # Enemy piece, valid for capture
                legal_moves.append((new_x, new_y))

        return legal_moves

