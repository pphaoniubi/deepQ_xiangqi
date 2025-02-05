def dessiner_plateau(board):
    """
    Affiche un plateau de Xiangqi (Chinese Chess) dans le terminal.
    """
    # En-tête des colonnes
    print("  1 2 3 4 5 6 7 8 9")
    
    # Dessiner chaque ligne du plateau
    for y in range(10):
        # Ajouter le numéro de la ligne
        row = f"{y+1} "
        for x in range(9):
            piece = board[y][x]
            # Si la case est vide, afficher "."
            if piece == 0:
                row += ". "
            else:
                # Afficher la première lettre de la pièce
                row += piece[0] + " "
        print(row)

# Exemple de configuration du plateau (Xiangqi)
# Utiliser 0 pour les cases vides et des symboles pour les pièces
plateau = [
    ["BRook", "BHorse", "BElephant", "BCannon", "BKing", "BCannon", "BElephant", "BHorse", "BRook"],
    ["BSoldier", "BSoldier", "BSoldier", "BSoldier", "BSoldier", "BSoldier", "BSoldier", "BSoldier", "BSoldier"],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ["RSoldier", "RSoldier", "RSoldier", "RSoldier", "RSoldier", "RSoldier", "RSoldier", "RSoldier", "RSoldier"],
    ["Rook", "RHorse", "RElephant", "RCannon", "RKing", "RCannon", "RElephant", "RHorse", "Rook"],
]

# Afficher le plateau
dessiner_plateau(plateau)
