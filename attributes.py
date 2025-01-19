
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

advisor_point_black = [(253 + 25, 55 + 25), (385 + 25, 55 + 25), (385 + 25, 187 + 25), (253 + 25, 187 + 25)]
advisor_center_black = (319 + 25, 121 + 25)

advisor_point_red = [(253 + 25, 649 + 25), (385 + 25, 649 + 25), (385 + 25, 517 + 25), (253 + 25, 517 + 25)]
advisor_center_red = (319 + 25, 583 + 25)


# Dictionnaire pour stocker les pièces et leurs rectangles
pieces = {}

grid_x = [55, 121, 187, 253, 319, 385, 451, 517, 583]
grid_y = [55, 121, 187, 253, 319, 385, 451, 517, 583, 649]


def is_piece_on_grid(piece_name, new_x, new_y):
    if piece_name.find("Black") != -1:
        for name, (image, rect) in pieces.items():  # Utiliser items() pour avoir la clé et la valeur
            if rect.x == new_x and rect.y == new_y \
            and piece_name != name:
                if name.find("Red") != -1:
                    del pieces[name]
                    break
                return True
            
    elif piece_name.find("Red") != -1:
        for name, (image, rect) in pieces.items():  # Utiliser items() pour avoir la clé et la valeur
            if rect.x == new_x and rect.y == new_y \
                and piece_name != name:
                if name.find("Black") != -1:
                    del pieces[name]
                    break
                return True
    return False

def find_closest_number(arr, target):
    # Initialize the closest value with a very high difference
    closest_value = arr[0]
    smallest_diff = abs(target - closest_value)

    # Loop through the array and find the closest value
    for num in arr:
        diff = abs(target - num)
        if diff < smallest_diff:
            closest_value = num
            smallest_diff = diff

    return closest_value

# Fonction pour recalculer les marges et dimensions de la grille
def recalculate_grid(window_width, window_height, gap):
    width = gap * 8
    height = gap * 9
    margin_x = (window_width - width) // 2
    margin_y = (window_height - height) // 2
    return width, height, margin_x, margin_y