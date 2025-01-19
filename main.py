import pygame
import sys
from attributes import *
from board_piece import *

# Initialisation de Pygame
pygame.init()

window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)

# Définir un titre pour la fenêtre
pygame.display.set_caption('xiangqi')



def draw_grid():
    # Dessiner 10 lignes horizontales
    for i in range(8):
        y = margin_y + (i + 1) * gap
        pygame.draw.line(window, BLACK, (margin_x, y), (margin_x + width, y), 2)  # Ligne horizontale

    # Dessiner 9 lignes verticales
    for j in range(7):
        x = margin_x + (j + 1) * gap
        pygame.draw.line(window, BLACK, (x, margin_y), (x, margin_y + height), 2)  # Ligne verticale
        
    # frontiere
    border_thickness = 2  # Épaisseur de la bordure
    pygame.draw.rect(window, BLACK, (margin_x, margin_y, width, height), border_thickness)

    for point in advisor_point_black:
        pygame.draw.line(window, BLACK, advisor_center_black, point, 2)

    for point in advisor_point_red:
        pygame.draw.line(window, BLACK, advisor_center_red, point, 2)
# Function to generate a 2D array of grid intersections
def gridpoint_coordinates(window_width, window_height, gap):
    # Recalculate grid dimensions and margins
    _, _, margin_x, margin_y = recalculate_grid(window_width, window_height, gap)  # Ignore width and height
    
    # Create the 2D array to store coordinates
    coordinates = []
    
    for row in range(10):  # 10 horizontal lines
        row_coords = []
        for col in range(9):  # 9 vertical lines
            # Calculate the x and y coordinates of each intersection
            x = margin_x + col * gap
            y = margin_y + row * gap
            row_coords.append((x, y))
        coordinates.append(row_coords)

    print(coordinates)
    return coordinates

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
gridpoint_coordinates(window_width, window_height, gap)
# Boucle principale du jeu
running = True
dragging_piece = None
dragging_offset_x = 0
dragging_offset_y = 0
selected_piece = None  # Keep track of the selected piece
initial_position = None  # To store the initial position of the piece

# Initializations (before the main loop)
selected_piece = None  # To store the currently selected piece

# Initializations
selected_piece = None  # To store the currently selected piece
dragging_piece = None  # To keep track of the piece being dragged
dragging_offset_x, dragging_offset_y = 0, 0  # Offset for dragging
initial_position = None  # To store the initial position of the piece

while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Quit the loop when closing the window

        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Detect if a piece is clicked
            mouse_pos = event.pos
            for piece_name, (image, rect) in pieces.items():
                if rect.collidepoint(mouse_pos):
                    center_x, center_y = rect.centerx, rect.centery
                    distance = ((mouse_pos[0] - center_x) ** 2 + (mouse_pos[1] - center_y) ** 2) ** 0.5
                    if distance <= 25:  # Click is within the piece radius
                        dragging_piece = piece_name  # Store the name of the piece being dragged
                        selected_piece = (piece_name, image, rect)  # Store the selected piece
                        initial_position = (rect.x, rect.y)  # Store the initial position
                        dragging_offset_x = mouse_pos[0] - rect.x
                        dragging_offset_y = mouse_pos[1] - rect.y
                        print(f"Sélection de {piece_name} à la position ({rect.x}, {rect.y})")
                        break  # Exit once a piece is selected

        elif event.type == pygame.MOUSEMOTION:
            # Drag the selected piece
            if dragging_piece:
                mouse_pos = event.pos
                piece_name, image, rect = selected_piece  # Unpack the selected piece
                new_x = mouse_pos[0] - dragging_offset_x
                new_y = mouse_pos[1] - dragging_offset_y

                # Temporarily update the rect to follow the mouse
                rect.x = new_x
                rect.y = new_y

        elif event.type == pygame.MOUSEBUTTONUP:
            # Place the dragged piece
            if dragging_piece:
                mouse_pos = event.pos
                piece_name, image, rect = selected_piece  # Unpack the selected piece
                new_x = mouse_pos[0] - dragging_offset_x
                new_y = mouse_pos[1] - dragging_offset_y
                new_x = find_closest_number(grid_x, new_x)
                new_y = find_closest_number(grid_y, new_y)

                print(f"Attempting to place {piece_name} at ({new_x}, {new_y})")

                # Check boundaries and validate the move
                if (0 <= new_x <= window_width - rect.width and
                    0 <= new_y <= window_height - rect.height):
                    rect.x = find_closest_number(grid_x, new_x)
                    rect.y = find_closest_number(grid_y, new_y)
                    if (all_moves(piece_name, new_x, new_y, initial_position[0], initial_position[1])
                    and is_piece_on_grid(piece_name, rect.x, rect.y) is False):
                        # placer-les sur la grille exactement
                        pieces[piece_name] = (image, rect)  # Update the piece's position
                        print(f"Placed {piece_name} at position ({rect.x}, {rect.y})")
                    else:
                        # Invalid move: return to initial position
                        rect.x, rect.y = initial_position
                        print(f"Invalid move for {piece_name}. Returning to initial position ({rect.x}, {rect.y}).")
                else:
                    # Out of bounds: return to initial position
                    rect.x, rect.y = initial_position
                    print(f"Out of bounds for {piece_name}. Returning to initial position ({rect.x}, {rect.y}).")

                # Reset dragging state
                dragging_piece = None
                selected_piece = None
                initial_position = None  # Reset initial position
    
    window.fill(WHITE)
    draw_grid()
    draw_pieces()
    # Mise à jour de l'écran
    pygame.display.flip()

# Quitter Pygame proprement
pygame.quit()
sys.exit()
