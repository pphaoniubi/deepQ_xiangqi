import pygame
from board_piece import change_sides
import game_state

gap = 66

width = gap * 8  # 9 colonnes verticales
height = gap * 9  # 10 lignes horizontales

window_width = width + 160  # Ajouter une marge pour le centrage horizontal
window_height = height + 160  # Ajouter une marge pour le centrage vertical

# Couleurs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

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

side = "Red"
red_win_count = 0
black_win_count = 0
red_time = 0
black_time = 0

running = True
dragging_piece = None
dragging_offset_x = 0
dragging_offset_y = 0
selected_piece = None
initial_position = None

selected_piece = None

# Initializations
selected_piece = None
dragging_piece = None
dragging_offset_x, dragging_offset_y = 0, 0
initial_position = None

