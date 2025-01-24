import pygame
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
                    return True
            
    elif piece_name.find("Red") != -1:
        for name, (image, rect) in pieces.items():  # Utiliser items() pour avoir la clé et la valeur
            if rect.x == new_x and rect.y == new_y \
                and piece_name != name:
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

def eliminate_piece(piece_name, x, y):
    # Parcourir toutes les pièces pour trouver celle à éliminer
    for name, (image, rect) in list(pieces.items()):
        if name == piece_name:
             continue
        elif rect.x == x and rect.y == y:
            del pieces[name]  # Supprimer la pièce trouvée
            print(f"Pièce '{name}' éliminée à la position ({x}, {y}).")
            return  # Sortir de la fonction après avoir supprimé la pièce
    print(f"Aucune pièce trouvée à la position ({x}, {y}).")

def get_color(piece_name, x, y):
    # Parcourir toutes les pièces pour trouver celle à éliminer
    for name, (image, rect) in list(pieces.items()):
        if name == piece_name:
            continue
        if rect.x == x and rect.y == y:
            if name.find("Black") != -1:
                return "Black"
            elif name.find("Red") != -1:
                return "Red"


def init_board():
    # Chariot
    B_Chariot_image = pygame.image.load("images/b_Chariot.png")  # Remplacez "piece.png" par le chemin de votre image
    B_Chariot_image = pygame.transform.scale(B_Chariot_image, (50, 50))

    for j in range(2):  # Pour chaque colonne
        x = margin_x + (j) * width  # Position X du croisement
        y = margin_y  # Position Y du croisement
        piece_rect = B_Chariot_image.get_rect(center=(x, y))  # Centrer l'image sur le croisement
        pieces[f"Black Chariot {j}"] = (B_Chariot_image, piece_rect)  # Dessiner l'image


    R_Chariot_image = pygame.image.load("images/r_Chariot.png")  # Remplacez "piece.png" par le chemin de votre image
    R_Chariot_image = pygame.transform.scale(R_Chariot_image, (50, 50))

    for j in range(2):  # Pour chaque colonne
        x = margin_x + (j) * width  # Position X du croisement
        y = margin_y + height # Position Y du croisement
        piece_rect = R_Chariot_image.get_rect(center=(x, y))  # Centrer l'image sur le croisement
        pieces[f"Red Chariot {j}"] = (R_Chariot_image, piece_rect)  # Dessiner l'image


    # Horse
    B_Horse_image = pygame.image.load("images/b_Horse.png")  # Remplacez "piece.png" par le chemin de votre image
    B_Horse_image = pygame.transform.scale(B_Horse_image, (50, 50))
    black_positions = [1, 7]
    for pos in black_positions:  
        x = margin_x + pos * gap
        y = margin_y
        piece_rect = B_Horse_image.get_rect(center=(x, y))
        pieces[f"Black Horse {pos}"] = (B_Horse_image, piece_rect)


    R_Horse_image = pygame.image.load("images/r_Horse.png")  # Remplacez "piece.png" par le chemin de votre image
    R_Horse_image = pygame.transform.scale(R_Horse_image, (50, 50))

    # Dessiner les pièces rouges en bas
    red_positions = [1, 7]
    for pos in red_positions:  
        x = margin_x + pos * gap
        y = margin_y + height
        piece_rect = R_Horse_image.get_rect(center=(x, y))
        pieces[f"Red Horse {pos}"] = (R_Horse_image, piece_rect)

    # Elephant
    B_Elephant_image = pygame.image.load("images/b_Elephant.png")
    B_Elephant_image = pygame.transform.scale(B_Elephant_image, (50, 50))
    black_positions = [2, 6]
    for pos in black_positions:  
        x = margin_x + pos * gap
        y = margin_y
        piece_rect = B_Elephant_image.get_rect(center=(x, y))
        pieces[f"Black Elephant {pos}"] = (B_Elephant_image, piece_rect)


    R_Elephant_image = pygame.image.load("images/r_Elephant.png")  # Remplacez "piece.png" par le chemin de votre image
    R_Elephant_image = pygame.transform.scale(R_Elephant_image, (50, 50))

    # Dessiner les pièces rouges en bas
    red_positions = [2, 6]
    for pos in red_positions:  
        x = margin_x + pos * gap
        y = margin_y + height
        piece_rect = R_Elephant_image.get_rect(center=(x, y))
        pieces[f"Red Elephant {pos}"] = (R_Elephant_image, piece_rect)


    # Advisor
    B_Advisor_image = pygame.image.load("images/b_Advisor.png")
    B_Advisor_image = pygame.transform.scale(B_Advisor_image, (50, 50))
    black_positions = [3, 5]
    for pos in black_positions:  
        x = margin_x + pos * gap
        y = margin_y
        piece_rect = B_Advisor_image.get_rect(center=(x, y))
        pieces[f"Black Advisor {pos}"] = (B_Advisor_image, piece_rect)


    R_Advisor_image = pygame.image.load("images/r_Advisor.png")  # Remplacez "piece.png" par le chemin de votre image
    R_Advisor_image = pygame.transform.scale(R_Advisor_image, (50, 50))

    # Dessiner les pièces rouges en bas
    red_positions = [3, 5]
    for pos in red_positions:  
        x = margin_x + pos * gap
        y = margin_y + height
        piece_rect = R_Advisor_image.get_rect(center=(x, y))
        pieces[f"Red Advisor {pos}"] = (R_Advisor_image, piece_rect)


    # General
    B_General_image = pygame.image.load("images/b_General.png")
    B_General_image = pygame.transform.scale(B_General_image, (50, 50))

    x = margin_x + 4 * gap
    y = margin_y
    piece_rect = B_General_image.get_rect(center=(x, y))
    pieces[f"Black General"] = (B_General_image, piece_rect)


    R_General_image = pygame.image.load("images/r_General.png")  # Remplacez "piece.png" par le chemin de votre image
    R_General_image = pygame.transform.scale(R_General_image, (50, 50))

    # Dessiner les pièces rouges en bas

    x = margin_x + 4 * gap
    y = margin_y + height
    piece_rect = R_General_image.get_rect(center=(x, y))
    pieces[f"Red General"] = (R_General_image, piece_rect)

    # Cannon
    B_Cannon_image = pygame.image.load("images/b_Cannon.png")
    B_Cannon_image = pygame.transform.scale(B_Cannon_image, (50, 50))
    black_positions = [1, 7]
    for pos in black_positions:  
        x = margin_x + pos * gap
        y = margin_y + 2 * gap
        piece_rect = B_Cannon_image.get_rect(center=(x, y))
        pieces[f"Black Cannon {pos}"] = (B_Cannon_image, piece_rect)


    R_Cannon_image = pygame.image.load("images/r_Cannon.png")  # Remplacez "piece.png" par le chemin de votre image
    R_Cannon_image = pygame.transform.scale(R_Cannon_image, (50, 50))

    # Dessiner les pièces rouges en bas
    red_positions = [1, 7]
    for pos in red_positions:  
        x = margin_x + pos * gap
        y = margin_y + 7 * gap
        piece_rect = R_Cannon_image.get_rect(center=(x, y))
        pieces[f"Red Cannon {pos}"] = (R_Cannon_image, piece_rect)

    # Soldier
    B_Soldier_image = pygame.image.load("images/b_Soldier.png")
    B_Soldier_image = pygame.transform.scale(B_Soldier_image, (50, 50))

    for j in range(9):  
        if j % 2 == 0:
            x = margin_x + j * gap
            y = margin_y + 3 * gap
            piece_rect = B_Soldier_image.get_rect(center=(x, y))
            pieces[f"Black Soldier {j}"] = (B_Soldier_image, piece_rect)


    R_Soldier_image = pygame.image.load("images/r_Soldier.png")  # Remplacez "piece.png" par le chemin de votre image
    R_Soldier_image = pygame.transform.scale(R_Soldier_image, (50, 50))

    # Dessiner les pièces rouges en bas
    for j in range(9):
        if j % 2 == 0:
            x = margin_x + j * gap
            y = margin_y + 6 * gap
            piece_rect = R_Soldier_image.get_rect(center=(x, y))
            pieces[f"Red Soldier {j}"] = (R_Soldier_image, piece_rect)

