import pygame
import sys

# Initialisation de Pygame
pygame.init()

# Définir l'espacement entre les lignes
gap = 66  # Espacement des lignes (vous pouvez ajuster cette valeur)

# Calcul automatique de la taille de la fenêtre
width = gap * 8  # 9 colonnes verticales
height = gap * 9  # 10 lignes horizontales

# Taille de la fenêtre
window_width = width + 160  # Ajouter une marge pour le centrage horizontal
window_height = height + 160  # Ajouter une marge pour le centrage vertical
window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)

# Définir un titre pour la fenêtre
pygame.display.set_caption('Grille 10x9')

# Couleurs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Calcul des marges pour centrer la grille
margin_x = (window_width - width) // 2
margin_y = (window_height - height) // 2

# Dictionnaire pour stocker les pièces et leurs rectangles
pieces = {}

# Fonction pour recalculer les marges et dimensions de la grille
def recalculate_grid(window_width, window_height, gap):
    width = gap * 8
    height = gap * 9
    margin_x = (window_width - width) // 2
    margin_y = (window_height - height) // 2
    return width, height, margin_x, margin_y

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

def draw_pieces():
    for image, rect in pieces.values():
        window.blit(image, rect)

init_board()
# Boucle principale du jeu
running = True
dragging_piece = None
dragging_offset_x = 0
dragging_offset_y = 0
while running:
    # Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Quitter la boucle lorsque l'on ferme la fenêtre

        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Détecter si une pièce a été cliquée
            mouse_pos = event.pos
            for i, (piece_name, (image, rect)) in enumerate(pieces.items()):
                if rect.collidepoint(mouse_pos):

                    center_x, center_y = rect.centerx, rect.centery
                    # La position de la pièce est déjà contenue dans rect.x et rect.y
                    distance = ((mouse_pos[0] - center_x) ** 2 + (mouse_pos[1] - center_y) ** 2) ** 0.5
                    if distance <= 25:  # Si le clic est dans le rayon de la pièce
                        dragging_piece = (piece_name, i)  # Stocker le nom de la pièce et son index
                        dragging_offset_x = mouse_pos[0] - center_x
                        dragging_offset_y = mouse_pos[1] - center_y
                        print(f"Sélection de {piece_name} à la position ({rect.x}, {rect.y})")
                        break  # Sortir de la boucle une fois qu'une pièce a été sélectionnée
                
        elif event.type == pygame.MOUSEMOTION:
            # Déplacer la pièce sélectionnée
            if dragging_piece:
                mouse_pos = event.pos
                type_piece, i = dragging_piece
                # Calculer la nouvelle position de la pièce
                new_x = mouse_pos[0] - dragging_offset_x - 25
                new_y = mouse_pos[1] - dragging_offset_y - 25
                print(mouse_pos[0], mouse_pos[1])
                # Récupérer l'image et le rect de la pièce
                image, rect = pieces[type_piece]

                # Mettre à jour les coordonnées du rect (déplacement de la pièce)
                rect.x = new_x
                rect.y = new_y

                # Réassigner l'image et la nouvelle position du rect dans le dictionnaire
                pieces[type_piece] = (image, rect)

        # Détecter le relâchement du bouton de la souris (fin du drag)
        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging_piece:
                # Vous pouvez ajouter ici la logique de validation de placement, si nécessaire
                print(f"Relâché la pièce {dragging_piece[0]} à la position finale.")
                
                # Mettre fin au "dragging" en réinitialisant la variable dragging_piece
                dragging_piece = None

        elif event.type == pygame.VIDEORESIZE:  # Détecter le redimensionnement
            window_width, window_height = event.w, event.h
            window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
            # Recalculer les dimensions de la grille
            width, height, margin_x, margin_y = recalculate_grid(window_width, window_height, gap)
    
    
    window.fill(WHITE)

    # Dessiner 10 lignes horizontales
    for i in range(8):
        y = margin_y + (i + 1) * gap
        pygame.draw.line(window, BLACK, (margin_x, y), (margin_x + width, y), 2)  # Ligne horizontale

    # Dessiner 9 lignes verticales
    for j in range(7):
        x = margin_x + (j + 1) * gap
        pygame.draw.line(window, BLACK, (x, margin_y), (x, margin_y + height), 2)  # Ligne verticale

    # Dessiner la frontière (rectangle autour de la grille)
    border_thickness = 2  # Épaisseur de la bordure
    pygame.draw.rect(window, BLACK, (margin_x, margin_y, width, height), border_thickness)


    draw_pieces()
    # Mise à jour de l'écran
    pygame.display.flip()

# Quitter Pygame proprement
pygame.quit()
sys.exit()
