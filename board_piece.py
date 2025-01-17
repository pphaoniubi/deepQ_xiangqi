from attributes import *

def all_moves(piece_name, new_x, new_y, init_x, init_y):

    if piece_name.startswith("Black Chariot") or piece_name.startswith("Red Chariot"):
        # to implement
        if ((abs(new_x - init_x) <= 10 and new_y != init_y) 
            or (new_x != init_x and abs(new_y - init_y) <= 10)):
            return True
        
    elif piece_name.startswith("Black Horse") or piece_name.startswith("Red Horse"):
        potential_moves = [
            (init_x + 2 * gap, init_y + gap),  # Bas-droite
            (init_x + 2 * gap, init_y - gap),  # Haut-droite
            (init_x - 2 * gap, init_y + gap),  # Bas-gauche
            (init_x - 2 * gap, init_y - gap),  # Haut-gauche
            (init_x + gap, init_y + 2 * gap),  # Droite-bas
            (init_x - gap, init_y + 2 * gap),  # Droite-haut
            (init_x + gap, init_y - 2 * gap),  # Gauche-bas
            (init_x - gap, init_y - 2 * gap)   # Gauche-haut
        ]
        valid_moves = []
        for nx, ny in potential_moves:
            # Vérifier si la position est dans les limites du plateau
            if 55 <= nx <= 583 and 55 <= ny <= 649:
                if nx == init_x + 2 * gap:
                    if is_piece_on_grid(piece_name, init_x + gap, init_y) is False:
                        valid_moves.append((nx, ny))
                elif  nx == init_x - 2 * gap:
                    if is_piece_on_grid(piece_name, init_x - gap, init_y) is False:
                        valid_moves.append((nx, ny))
                elif ny == init_y + 2 * gap:
                    if is_piece_on_grid(piece_name, init_x, init_y + gap) is False:
                        valid_moves.append((nx, ny))
                elif ny == init_y - 2 * gap:
                    if is_piece_on_grid(piece_name, init_x, init_y - gap) is False:
                        valid_moves.append((nx, ny))
        for nx, ny in valid_moves:
            if new_x == nx and new_y == ny:
                # to implement
                return True
    
    elif piece_name.startswith("Black Elephant") or piece_name.startswith("Red Elephant"):
        # to implement
            return True
    
    elif piece_name.startswith("Black Advisor") or piece_name.startswith("Red Advisor"):
        # to implement
            return True
    
    elif piece_name.startswith("Black General") or piece_name.startswith("Red General"):
        # to implement
            return True
    
    elif piece_name.startswith("Black Cannon") or piece_name.startswith("Red Cannon"):
        # to implement
            return True
    
    elif piece_name.startswith("Black Soldier") or piece_name.startswith("Red Soldier"):
        # to implement
            return False

    return False