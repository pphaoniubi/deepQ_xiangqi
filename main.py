import pygame
import sys

# Initialisation de Pygame
pygame.init()

# Définir l'espacement entre les lignes
gap = 80  # Espacement des lignes (vous pouvez ajuster cette valeur)

# Calcul automatique de la taille de la fenêtre
width = gap * 9 + 20  # 9 colonnes verticales
height = gap * 10 + 20  # 10 lignes horizontales

# Taille de la fenêtre
window_width = width + 40  # Ajouter une marge pour le centrage horizontal
window_height = height + 40  # Ajouter une marge pour le centrage vertical
window = pygame.display.set_mode((window_width, window_height))

# Définir un titre pour la fenêtre
pygame.display.set_caption('Grille 10x9')

# Couleurs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BORDER_COLOR = (255, 0, 0)

# Chess piece symbols or images
# Here we just use basic symbols for pieces, like 'R' for rook, 'N' for knight, etc.
# You can replace these with images if needed
WHITE_PAWN = '♙'
WHITE_ROOK = '♖'
WHITE_KNIGHT = '♘'
WHITE_BISHOP = '♗'
WHITE_QUEEN = '♕'
WHITE_KING = '♔'

BLACK_PAWN = '♟'
BLACK_ROOK = '♜'
BLACK_KNIGHT = '♞'
BLACK_BISHOP = '♝'
BLACK_QUEEN = '♛'
BLACK_KING = '♚'

# Chess pieces initial positions (adjusted for 10x9 grid)
white_pieces = [
    (0, 0, WHITE_ROOK), (1, 0, WHITE_KNIGHT), (2, 0, WHITE_BISHOP),
    (3, 0, WHITE_QUEEN), (4, 0, WHITE_KING), (5, 0, WHITE_BISHOP),
    (6, 0, WHITE_KNIGHT), (7, 0, WHITE_ROOK),
    (0, 1, WHITE_PAWN), (1, 1, WHITE_PAWN), (2, 1, WHITE_PAWN),
    (3, 1, WHITE_PAWN), (4, 1, WHITE_PAWN), (5, 1, WHITE_PAWN),
    (6, 1, WHITE_PAWN), (7, 1, WHITE_PAWN)
]

black_pieces = [
    (0, 8, BLACK_ROOK), (1, 8, BLACK_KNIGHT), (2, 8, BLACK_BISHOP),
    (3, 8, BLACK_QUEEN), (4, 8, BLACK_KING), (5, 8, BLACK_BISHOP),
    (6, 8, BLACK_KNIGHT), (7, 8, BLACK_ROOK),
    (0, 9, BLACK_PAWN), (1, 9, BLACK_PAWN), (2, 9, BLACK_PAWN),
    (3, 9, BLACK_PAWN), (4, 9, BLACK_PAWN), (5, 9, BLACK_PAWN),
    (6, 9, BLACK_PAWN), (7, 9, BLACK_PAWN)
]

# Function to draw text in the grid
def draw_piece(piece, x, y):
    font = pygame.font.SysFont("Arial", 30)
    text = font.render(piece, True, BLACK)
    window.blit(text, (margin_x + x * gap + gap // 4, margin_y + y * gap + gap // 4))

# Calcul des marges pour centrer la grille
margin_x = (window_width - width) // 2
margin_y = (window_height - height) // 2

# Boucle principale du jeu
running = True
while running:
    # Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Quitter la boucle lorsque l'on ferme la fenêtre
    
    # Remplir l'arrière-plan (blanc)
    window.fill(WHITE)

    # Dessiner 10 lignes horizontales
    for i in range(10):
        y = margin_y + (i + 1) * gap  # Position verticale des lignes horizontales
        pygame.draw.line(window, BLACK, (margin_x, y), (margin_x + width, y), 2)  # Ligne horizontale

    # Dessiner 9 lignes verticales
    for j in range(9):
        x = margin_x + (j + 1) * gap  # Position horizontale des lignes verticales
        pygame.draw.line(window, BLACK, (x, margin_y), (x, margin_y + height), 2)  # Ligne verticale

    # Dessiner la frontière (rectangle autour de la grille)
    border_thickness = 1  # Épaisseur de la bordure
    pygame.draw.rect(window, BORDER_COLOR, (margin_x, margin_y, width, height), border_thickness)

    # Draw pieces on the grid
    for x, y, piece in white_pieces + black_pieces:
        draw_piece(piece, x, y)

    # Mise à jour de l'écran
    pygame.display.flip()

# Quitter Pygame proprement
pygame.quit()
sys.exit()
