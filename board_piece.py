from attributes import *

def all_moves(piece_name, new_x, new_y, init_x, init_y):

    if piece_name.startswith("Black Chariot") or piece_name.startswith("Red Chariot"):
        # to implement
        num_gap_x = 0
        num_gap_y = 0
        if (new_x != init_x and new_y == init_y):
            num_gap_x = abs(new_x - init_x) / gap
        elif (new_x == init_x and new_y != init_y):
            num_gap_y = abs(new_y - init_y) / gap
        potential_moves = [
                (init_x, init_y + gap * num_gap_y),  # En avant
                (init_x, init_y - gap * num_gap_y),  # En arrière
                (init_x - num_gap_x * gap, init_y),  # À gauche
                (init_x + num_gap_x * gap, init_y),  # À droite
            ]
        
        valid_moves = []

        for nx, ny in potential_moves:
            if is_piece_on_grid(piece_name, new_x, new_y) is False:
                        valid_moves.append((nx, ny))

        for nx, ny in valid_moves:
            if new_x == nx and new_y == ny:
                # to implement
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
        potential_moves = [
            (init_x + 2 * gap, init_y + 2 * gap),
            (init_x + 2 * gap, init_y - 2 * gap),
            (init_x - 2 * gap, init_y + 2 * gap),
            (init_x - 2 * gap, init_y - 2 * gap),
        ]

        valid_moves = []
        for nx, ny in potential_moves:
            # Vérifier si la position est dans les limites du plateau
            if piece_name.startswith("Black Elephant"):
                if 55 <= nx <= 583 and 55 <= ny <= 319:
                    if is_piece_on_grid(piece_name, new_x, new_y) is False:
                        valid_moves.append((nx, ny))

            if piece_name.startswith("Red Elephant"):
                if 55 <= nx <= 583 and 385 <= ny <= 649:
                    if is_piece_on_grid(piece_name, new_x, new_y) is False:
                        valid_moves.append((nx, ny))

        for nx, ny in valid_moves:
            if new_x == nx and new_y == ny:
                # to implement
                return True

    
    elif piece_name.startswith("Black Advisor") or piece_name.startswith("Red Advisor"):
        # to implement
        potential_moves = [
            (init_x + gap, init_y + gap),
            (init_x + gap, init_y - gap),
            (init_x - gap, init_y + gap),
            (init_x - gap, init_y - gap),
        ]
        
        valid_moves = []
        for nx, ny in potential_moves:
            # Vérifier si la position est dans les limites du plateau
            if piece_name.startswith("Black Advisor"):
                if 253 <= nx <= 385 and 55 <= ny <= 187:
                    if is_piece_on_grid(piece_name, new_x, new_y) is False:
                        valid_moves.append((nx, ny))

            if piece_name.startswith("Red Advisor"):
                if 253 <= nx <= 385 and 517 <= ny <= 649:
                    if is_piece_on_grid(piece_name, new_x, new_y) is False:
                        valid_moves.append((nx, ny))

        for nx, ny in valid_moves:
            if new_x == nx and new_y == ny:
                # to implement
                return True
    
    elif piece_name.startswith("Black General") or piece_name.startswith("Red General"):
        # to implement
        potential_moves = [
            (init_x + gap, init_y),
            (init_x - gap, init_y),
            (init_x, init_y + gap),
            (init_x, init_y - gap),
        ]
        
        valid_moves = []
        for nx, ny in potential_moves:
            # Vérifier si la position est dans les limites du plateau
            if piece_name.startswith("Black General"):
                if 253 <= nx <= 385 and 55 <= ny <= 187:
                    if is_piece_on_grid(piece_name, new_x, new_y) is False:
                        valid_moves.append((nx, ny))

            if piece_name.startswith("Red General"):
                if 253 <= nx <= 385 and 517 <= ny <= 649:
                    if is_piece_on_grid(piece_name, new_x, new_y) is False:
                        valid_moves.append((nx, ny))

        for nx, ny in valid_moves:
            if new_x == nx and new_y == ny:
                # to implement
                return True
    
    elif piece_name.startswith("Black Cannon") or piece_name.startswith("Red Cannon"):
        # to implement
        num_gap_x = 0
        num_gap_y = 0
        if (new_x != init_x and new_y == init_y):
            num_gap_x = abs(new_x - init_x) / gap
        elif (new_x == init_x and new_y != init_y):
            num_gap_y = abs(new_y - init_y) / gap
        potential_moves = [
                (init_x, init_y + gap * num_gap_y),  # En avant
                (init_x, init_y - gap * num_gap_y),  # En arrière
                (init_x - num_gap_x * gap, init_y),  # À gauche
                (init_x + num_gap_x * gap, init_y),  # À droite
            ]
        
        
        valid_moves = []

        for nx, ny in potential_moves:
            if is_piece_on_grid(piece_name, new_x, new_y) is False:
                        valid_moves.append((nx, ny))

        for nx, ny in valid_moves:
            if new_x == nx and new_y == ny:
                # to implement
                return True
    
    elif piece_name.startswith("Black Soldier"): 
        valid_moves = []

        # Vérifier si la pièce a traversé le fleuve (ici, y = 5 est la ligne du fleuve)
        has_crossed_river = (init_y >= 5 * gap + 55)

        # Mouvement vers l'avant (avant la ligne du fleuve ou après)
        if has_crossed_river:
            # Après avoir traversé le fleuve, on peut aller à gauche, à droite ou en avant
            potential_moves = [
                (init_x, init_y + gap),  # En avant
                (init_x - gap, init_y),  # À gauche
                (init_x + gap, init_y),  # À droite
            ]
        else:
            # Avant le fleuve, on ne peut aller que vers l'avant
            potential_moves = [
                (init_x, init_y + gap),  # En avant
            ]
                # Vérifier les déplacements valides dans les limites du plateau
        for nx, ny in potential_moves:
            if 55 <= nx <= 583 and 55 <= ny <= 649:
                # Ajouter le mouvement à la liste des mouvements valides
                valid_moves.append((nx, ny))

        for nx, ny in valid_moves:
            if new_x == nx and new_y == ny:
                # to implement
                return True
    elif piece_name.startswith("Red Soldier"):
        valid_moves = []

        # Vérifier si la pièce a traversé le fleuve (ici, y = 5 est la ligne du fleuve)
        has_crossed_river = (init_y <= 4 * gap + 55)

        # Mouvement vers l'avant (avant la ligne du fleuve ou après)
        if has_crossed_river:
            # Après avoir traversé le fleuve, on peut aller à gauche, à droite ou en avant
            potential_moves = [
                (init_x, init_y - gap),  # En avant
                (init_x - gap, init_y),  # À gauche
                (init_x + gap, init_y),  # À droite
            ]
        else:
            # Avant le fleuve, on ne peut aller que vers l'avant
            potential_moves = [
                (init_x, init_y - gap),  # En avant
            ]
                # Vérifier les déplacements valides dans les limites du plateau
        for nx, ny in potential_moves:
            if 55 <= nx <= 583 and 55 <= ny <= 649:
                # Ajouter le mouvement à la liste des mouvements valides
                valid_moves.append((nx, ny))

        for nx, ny in valid_moves:
            if new_x == nx and new_y == ny:
                # to implement
                return True
    return False