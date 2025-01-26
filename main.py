import pygame
import sys
from attributes import *
from board_piece import *

# Initialisation de Pygame
pygame.init()

window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)


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
    border_thickness = 2
    pygame.draw.rect(window, BLACK, (margin_x, margin_y, width, height), border_thickness)

    for point in advisor_point_black:
        pygame.draw.line(window, BLACK, advisor_center_black, point, 2)

    for point in advisor_point_red:
        pygame.draw.line(window, BLACK, advisor_center_red, point, 2)

def draw_pieces():
    for image, rect in pieces.values():
        window.blit(image, rect)

init_board()
def pieces_to_board(pieces):
    board = [[0]*9 for i in range(10)]
    for name, (image, rect) in pieces.items():
        i = int((rect.x - 55) / gap)
        j = int((rect.y - 55) / gap)
        if name.find("Black") != -1:
            board[j][i] = 1
        elif name.find("Red") != -1:
            board[j][i] = -1
    board_flat = []
    for row in board:
        for cross in row:
            board_flat.append(cross)
    print(board_flat)
    return board

pieces_to_board(pieces)
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            for piece_name, (image, rect) in pieces.items():
                if rect.collidepoint(mouse_pos):

                    if piece_name.find("Red") != -1 and side == "Black":
                        print("Vous ne pouvez pas sélectionner les pièces de l'adversaire.")
                        break  # Ignore cette pièce et passe à la suivante
                    elif piece_name.find("Black") != -1 and side == "Red":
                        print("Vous ne pouvez pas sélectionner les pièces de l'adversaire.")
                        break  # Ignore cette pièce et passe à la suivante

                    center_x, center_y = rect.centerx, rect.centery
                    distance = ((mouse_pos[0] - center_x) ** 2 + (mouse_pos[1] - center_y) ** 2) ** 0.5
                    if distance <= 25:
                        dragging_piece = piece_name
                        selected_piece = (piece_name, image, rect)
                        initial_position = (rect.x, rect.y)
                        dragging_offset_x = mouse_pos[0] - rect.x
                        dragging_offset_y = mouse_pos[1] - rect.y
                        print(f"Sélection de {piece_name} à la position ({rect.x}, {rect.y})")
                        break 
        elif event.type == pygame.MOUSEMOTION:

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
                            side = change_sides(side)
                            print(f"Placed {piece_name} at position ({rect.x}, {rect.y})")
                        elif is_piece_on_grid(piece_name, rect.x, rect.y) is True:
                            if eliminate_piece(piece_name, rect.x, rect.y) is False:
                                rect.x, rect.y = initial_position
                                side = change_sides(side)
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
                initial_position = None
    
    window.fill(WHITE)
    draw_grid()
    draw_pieces()
    pygame.display.flip()
    if is_winning() == "Red wins":
        red_win_count += 1
        pieces.clear()
        init_board()
        side = "Red"
        print("Red wins")
    elif is_winning() == "Black wins":
        black_win_count += 1
        pieces.clear()
        init_board()
        side = "Red"
        print("Black wins")

# Quitter Pygame proprement
pygame.quit()
sys.exit()
