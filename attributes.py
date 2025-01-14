
# Définir l'espacement entre les lignes
gap = 66  # Espacement des lignes (vous pouvez ajuster cette valeur)

# Calcul automatique de la taille de la fenêtre
width = gap * 8  # 9 colonnes verticales
height = gap * 9  # 10 lignes horizontales

# Taille de la fenêtre
window_width = width + 160  # Ajouter une marge pour le centrage horizontal
window_height = height + 160  # Ajouter une marge pour le centrage vertical

# Couleurs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


# Calcul des marges pour centrer la grille
margin_x = (window_width - width) // 2
margin_y = (window_height - height) // 2


# Dictionnaire pour stocker les pièces et leurs rectangles
pieces = {}