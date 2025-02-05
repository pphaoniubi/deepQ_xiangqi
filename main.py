def dessiner_plateau(board):
    """
    Affiche un plateau de Xiangqi (Chinese Chess) dans le terminal.
    """
    # En-tête des colonnes
    print("   1   2   3   4   5   6   7   8   9")
    
    # Dessiner chaque ligne du plateau
    for y in range(10):
        # Ajouter le numéro de la ligne
        row = f"{y+1} "
        for x in range(9):
            piece = board[y][x]
            # Si la case est vide, afficher "."
            if piece == 0:
                row += " .  "
            elif 0 < piece < 10:
                # Afficher la première lettre de la pièce
                row += f'+{piece}  '
            elif piece >= 10:
                # Afficher la première lettre de la pièce
                row += f'+{piece} '
            elif -10 < piece < 0:
                # Afficher la première lettre de la pièce
                row += f'{piece}  '
            elif piece <= -10:
                # Afficher la première lettre de la pièce
                row += f'{piece} '
        print(row)

# Exemple de configuration du plateau (Xiangqi)
# Utiliser 0 pour les cases vides et des symboles pour les pièces
plateau_init = [
    [-1, -2, -3, -4, -5, -6, -7, -8, -9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -10, 0, 0, 0, 0, 0, -11, 0],
    [-12, 0, -13, 0, -14, 0, -15, 0, -16],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [12, 0, 13, 0, 14, 0, 15, 0, 16],
    [0, 10, 0, 0, 0, 0, 0, 11, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
]

# Afficher le plateau
dessiner_plateau(plateau_init)
