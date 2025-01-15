import pygame
import sys
from attributes import *
from board_piece import *

# Initialisation de Pygame
pygame.init()

window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)

# Définir un titre pour la fenêtre
pygame.display.set_caption('Grille 10x9')

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
# Board dimensions
print("Board params (width, heigth, margin_x, margin_y): ", width, height, margin_x, margin_y)
# Boucle principale du jeu
running = True
dragging_piece = None
dragging_offset_x = 0
dragging_offset_y = 0
selected_piece = None  # Keep track of the selected piece

while running:
    # Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Quitter la boucle lorsque l'on ferme la fenêtre

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos

            if selected_piece is None:  # First click: Select a piece
                for i, (piece_name, (image, rect)) in enumerate(pieces.items()):
                    if rect.collidepoint(mouse_pos):  # Check if the click is on a piece
                        center_x, center_y = rect.centerx, rect.centery
                        distance = ((mouse_pos[0] - center_x) ** 2 + (mouse_pos[1] - center_y) ** 2) ** 0.5
                        if distance <= 25:  # Ensure the click is within the piece's radius
                            selected_piece = (piece_name, image, rect)  # Store the selected piece
                            print(f"Selected {piece_name} at position ({rect.x}, {rect.y})")
                            break  # Exit the loop once a piece is selected

            elif selected_piece is not None:  # Second click: Place the piece
                # Unpack the selected piece
                piece_name, image, rect = selected_piece

                # Optional: Check if the placement is valid (within grid bounds, no overlap)
                # For example, check if the piece is within the grid boundaries:
                if (0 <= rect.x <= window_width - rect.width and
                    0 <= rect.y <= window_height - rect.height):
                    #Check for validity of position and possibly update var valid-placement (to implement)
                    if (True): #TO REPLACE WITH if (all_moves(piece_name, rect.x, rect.y, mouse_pos[0] - rect.width // 2, mouse_pos[1] - rect.height // 2)):
                        # Move the piece to the new position (center the piece on the click)
                        rect.x = mouse_pos[0] - rect.width // 2
                        rect.y = mouse_pos[1] - rect.height // 2
                        # Update the piece's position in the dictionary
                        pieces[piece_name] = (image, rect)
                        print(f"Placed {piece_name} at position ({rect.x}, {rect.y})")
                    else:
                        print(f"Invalid placement for {piece_name}. Invalid move!")
                else:
                    print(f"Invalid placement for {piece_name}. Out of bounds!")

                # Reset the selected piece after placing it
                selected_piece = None

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
