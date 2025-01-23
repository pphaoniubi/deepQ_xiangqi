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
                    if (is_move_valid(piece_name, new_x, new_y, initial_position[0], initial_position[1])):
                        if is_piece_on_grid(piece_name, rect.x, rect.y) is False:
                            # placer-les sur la grille exactement
                            pieces[piece_name] = (image, rect)  # Update the piece's position
                            print(f"Placed {piece_name} at position ({rect.x}, {rect.y})")
                        elif is_piece_on_grid(piece_name, rect.x, rect.y) is True:
                            eliminate_piece(piece_name, rect.x, rect.y)
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
