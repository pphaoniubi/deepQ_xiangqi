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

piece_image = pygame.image.load("images/b_Advisor.png")  # Remplacez "piece.png" par le chemin de votre image
piece_image = pygame.transform.scale(piece_image, (40, 40))

# Fonction pour recalculer les marges et dimensions de la grille
def recalculate_grid(window_width, window_height, gap):
    width = gap * 8
    height = gap * 9
    margin_x = (window_width - width) // 2
    margin_y = (window_height - height) // 2
    return width, height, margin_x, margin_y

def init_board():
    # Dessiner les pièces aux croisements
    for i in range(10):  # Pour chaque ligne
        for j in range(9):  # Pour chaque colonne
            x = margin_x + (j) * gap  # Position X du croisement
            y = margin_y + (i) * gap  # Position Y du croisement
            piece_rect = piece_image.get_rect(center=(x, y))  # Centrer l'image sur le croisement
            window.blit(piece_image, piece_rect)  # Dessiner l'image



# Boucle principale du jeu
running = True
while running:
    # Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Quitter la boucle lorsque l'on ferme la fenêtre

        elif event.type == pygame.VIDEORESIZE:  # Détecter le redimensionnement
            window_width, window_height = event.w, event.h
            window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
            # Recalculer les dimensions de la grille
            width, height, margin_x, margin_y = recalculate_grid(window_width, window_height, gap)
    
    
    # Remplir l'arrière-plan (noir)
    window.fill(WHITE)

    # Dessiner 10 lignes horizontales
    for i in range(8):
        y = margin_y + (i + 1) * gap  # Position verticale des lignes horizontales
        pygame.draw.line(window, BLACK, (margin_x, y), (margin_x + width, y), 2)  # Ligne horizontale

    # Dessiner 9 lignes verticales
    for j in range(7):
        x = margin_x + (j + 1) * gap  # Position horizontale des lignes verticales
        pygame.draw.line(window, BLACK, (x, margin_y), (x, margin_y + height), 2)  # Ligne verticale

    # Dessiner la frontière (rectangle autour de la grille)
    border_thickness = 2  # Épaisseur de la bordure
    pygame.draw.rect(window, BLACK, (margin_x, margin_y, width, height), border_thickness)


    init_board()

    # Mise à jour de l'écran
    pygame.display.flip()

# Quitter Pygame proprement
pygame.quit()
sys.exit()
