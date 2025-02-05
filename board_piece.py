from attributes import *
import deepQ

def change_sides(side):
    if side == "Red":
        side = side.replace("Red", "Black")
    elif side == "Black":
        side = side.replace("Black", "Red")
    
    return side


def is_winning():
    if "Black General" not in pieces:
        return "Red wins"
    elif "Red General" not in pieces:
        return "Black wins"
    else:
        return "Game continues"
    

def make_move(piece_name, new_x, new_y):
    for name, (image, rect) in pieces.items():
       if name == piece_name:
            rect.x = new_x
            rect.y = new_y
            break
       

def get_legal_moves(piece_name, init_x, init_y, board):

    if piece_name.startswith("Black Chariot"):
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
                    elif board[y][x] < 0 and piece_name < 0:  # Friendly piece
                        break
                    elif board[y][x] > 0:  # Enemy piece, valid for capture
                        legal_moves.append((x, y))
                        break  # Stop after capturing
                    
            return legal_moves

            
    elif piece_name.startswith("Red Chariot"):
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
                elif board[y][x] > 0 and piece_name > 0:  # Friendly piece
                    break
                elif board[y][x] < 0:  # Enemy piece, valid for capture
                    legal_moves.append((x, y))
                    break  # Stop after capturing
                
        return legal_moves
            
    elif piece_name.startswith("Black Horse"):
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
                    if board[new_y][new_x] == 0 or board[new_y][new_x] > 0:
                        legal_moves.append((new_x, new_y))

        return legal_moves
            
    elif piece_name.startswith("Red Horse"): 
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
                    if board[new_y][new_x] == 0 or board[new_y][new_x] < 0:
                        legal_moves.append((new_x, new_y))

        return legal_moves
            
    elif piece_name.startswith("Black Elephant"):
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
            if 55 <= new_x <= 583 and 385 <= new_y <= 649:
                if (new_y <= 4) or (new_y >= 5):
                    # Vérifier si la "jambe" est bloquée
                    block_x, block_y = block_positions[(dx, dy)]
                    if board[block_y][block_x] == 0:  # Pas d'obstacle
                        # Vérifier que la case de destination est vide ou contient un ennemi
                        if board[new_y][new_x] == 0 or board[new_y][new_x] > 0:
                            legal_moves.append((new_x, new_y))

        return legal_moves

    elif piece_name.startswith("Red Elephant"):
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
            if 55 <= new_x <= 583 and 385 <= new_y <= 649:
                if (new_y <= 4) or (new_y >= 5):
                    # Vérifier si la "jambe" est bloquée
                    block_x, block_y = block_positions[(dx, dy)]
                    if board[block_y][block_x] == 0:  # Pas d'obstacle
                        # Vérifier que la case de destination est vide ou contient un ennemi
                        if board[new_y][new_x] == 0 or board[new_y][new_x] > 0:
                            legal_moves.append((new_x, new_y))

        return legal_moves
            
    elif piece_name.startswith("Black Advisor"):
        legal_moves = []

        # Liste des 8 mouvements possibles
        advisor_moves = [
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        for dx, dy in advisor_moves:
            new_x, new_y = init_x + dx, init_y + dy

            # Vérifier que la destination est sur le plateau
            if 3 <= new_x <= 5 and 0 <= new_y <= 2:
                # Vérifier que la case de destination est vide ou contient un ennemi
                if board[new_y][new_x] == 0 or board[new_y][new_x] > 0:
                    legal_moves.append((new_x, new_y))

        return legal_moves
    
    elif piece_name.startswith("Red Advisor"):
        legal_moves = []

        # Liste des 8 mouvements possibles
        advisor_moves = [
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        for dx, dy in advisor_moves:
            new_x, new_y = init_x + dx, init_y + dy

            # Vérifier que la destination est sur le plateau
            if 3 <= new_x <= 5 and 7 <= new_y <= 9:
                # Vérifier que la case de destination est vide ou contient un ennemi
                if board[new_y][new_x] == 0 or board[new_y][new_x] < 0:
                    legal_moves.append((new_x, new_y))

        return legal_moves

    elif piece_name.startswith("Black General"):
        legal_moves = []

        # Liste des 8 mouvements possibles
        general_moves = [
            (1, 0), (0, 1), (-1, 0), (0, -1)
        ]

        for dx, dy in general_moves:
            new_x, new_y = init_x + dx, init_y + dy

            # Vérifier que la destination est sur le plateau
            if 3 <= new_x <= 5 and 0 <= new_y <= 2:
                # Vérifier que la case de destination est vide ou contient un ennemi
                if board[new_y][new_x] == 0 or board[new_y][new_x] > 0:
                    legal_moves.append((new_x, new_y))

        return legal_moves
    
    elif  piece_name.startswith("Red General"):
        legal_moves = []

        # Liste des 8 mouvements possibles
        general_moves = [
            (1, 0), (0, 1), (-1, 0), (0, -1)
        ]

        for dx, dy in general_moves:
            new_x, new_y = init_x + dx, init_y + dy

            # Vérifier que la destination est sur le plateau
            if 3 <= new_x <= 5 and 7 <= new_y <= 9:
                # Vérifier que la case de destination est vide ou contient un ennemi
                if board[new_y][new_x] == 0 or board[new_y][new_x] < 0:
                    legal_moves.append((new_x, new_y))

        return legal_moves
            
    elif piece_name.startswith("Black Cannon"):
        legal_moves = []

        if piece_name.startswith("Black Cannon") or piece_name.startswith("Red Cannon"):
            # Directions de mouvement (horizontalement et verticalement)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Gauche, Droite, Haut, Bas
            
            for dx, dy in directions:
                x, y = init_x, init_y
                passed_piece = False  # Indicateur pour savoir si une pièce a été rencontrée

                while True:
                    x += dx
                    y += dy

                    # Vérifier si la position est hors limites
                    if not (0 <= x < 9 and 0 <= y < 10):
                        break  # Stop si hors du plateau
                    
                    # Vérifier les cases
                    if board[y][x] == 0:  # Case vide
                        if passed_piece:  # Si une pièce alliée a été traversée, la capture est possible
                            legal_moves.append((x, y))
                    elif board[y][x].startswith("Black") or board[y][x].startswith("Red"):  # Pièce alliée
                        if not passed_piece:  # Une pièce alliée bloque la route, on ne peut pas passer
                            break
                        else:  # Si une pièce a été passée, on peut sauter
                            continue
                    else:  # Pièce ennemie
                        if passed_piece:  # La pièce ennemie peut être capturée
                            legal_moves.append((x, y))
                        break  # Une fois qu'une pièce ennemie est capturée, on arrête le mouvement

                    # Si on rencontre une pièce alliée, on la marque comme un "passage"
                    if board[y][x] != 0:
                        passed_piece = True
            
        return legal_moves
    
    elif piece_name.startswith("Red Cannon"):
        piece_hit_count = 0
        num_gap_x = 0
        num_gap_y = 0
        if (new_x != init_x and new_y == init_y): # Déplacement horizontal
            num_gap_x = abs(new_x - init_x) / gap

            temp_pos_x = init_x
            for i in range(round(num_gap_x)):
                if new_x > init_x:
                    temp_pos_x += gap
                else:
                    temp_pos_x -= gap

                if is_piece_on_grid(piece_name, temp_pos_x, init_y):
                    piece_hit_count += 1

                    if piece_hit_count == 2 and get_color(piece_name, new_x, new_y) == "Black" \
                        and temp_pos_x == new_x and init_y == new_y:
                        return True
                    
            if piece_hit_count == 0:
                return True
            elif piece_hit_count == 1:
                return False
            elif piece_hit_count == 2 and is_piece_on_grid(piece_name, temp_pos_x, init_y) is False:
                return False
            return False

        elif (new_x == init_x and new_y != init_y):  # Déplacement vertical
            num_gap_y = abs(new_y - init_y) / gap

            temp_pos_y = init_y
            for i in range(round(num_gap_y)):
                if new_y > init_y:
                    temp_pos_y += gap
                else:
                    temp_pos_y -= gap

                if is_piece_on_grid(piece_name, init_x, temp_pos_y):
                    piece_hit_count += 1
                    if piece_hit_count == 2 and get_color(piece_name, new_x, new_y) == "Black" \
                        and init_x == new_x and temp_pos_y == new_y:
                        return True

            if piece_hit_count == 0:
                return True
            elif piece_hit_count == 1:
                return False
            elif piece_hit_count == 2 and is_piece_on_grid(piece_name, init_x, temp_pos_y) is False:
                return False
            return False

    elif piece_name.startswith("Black Soldier"): 
        valid_moves = []

        has_crossed_river = (init_y >= 5 * gap + 55)

        if has_crossed_river:
            potential_moves = [
                (init_x, init_y + gap),
                (init_x - gap, init_y),
                (init_x + gap, init_y),
            ]
        else:

            potential_moves = [
                (init_x, init_y + gap),
            ]
        for nx, ny in potential_moves:
            if 55 <= nx <= 583 and 55 <= ny <= 649:
                valid_moves.append((nx, ny))

        for nx, ny in valid_moves:
            if new_x == nx and new_y == ny:
                # to implement
                return True
    elif piece_name.startswith("Red Soldier"):
        valid_moves = []

        has_crossed_river = (init_y <= 4 * gap + 55)

        if has_crossed_river:
            potential_moves = [
                (init_x, init_y - gap),
                (init_x - gap, init_y),
                (init_x + gap, init_y),
            ]
        else:
            potential_moves = [
                (init_x, init_y - gap),
            ]

        for nx, ny in potential_moves:
            if 55 <= nx <= 583 and 55 <= ny <= 649:
                valid_moves.append((nx, ny))

        for nx, ny in valid_moves:
            if new_x == nx and new_y == ny:
                return True
    return False