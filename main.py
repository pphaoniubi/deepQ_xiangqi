import pygame
import sys

# Initialisation de Pygame
pygame.init()

# Définir l'espacement entre les lignes
gap = 80  # Espacement des lignes (vous pouvez ajuster cette valeur)

# Calcul automatique de la taille de la fenêtre
width = gap * 9 + 0.5  # 9 colonnes verticales
height = gap * 10 + 0.5  # 10 lignes horizontales

window = pygame.display.set_mode((width, height))

# Définir un titre pour la fenêtre
pygame.display.set_caption('Grille 10x9')

# Couleurs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BORDER_COLOR = (255, 0, 0)

# Boucle principale du jeu
running = True
while running:
    # Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Quitter la boucle lorsque l'on ferme la fenêtre
    
    # Remplir l'arrière-plan (noir)
    window.fill(WHITE)

    # Dessiner 10 lignes horizontales
    for i in range(10):
        y = (i + 1) * gap  # Position verticale des lignes horizontales
        pygame.draw.line(window, BLACK, (0, y), (width, y), 2)  # Ligne horizontale

    # Dessiner 9 lignes verticales
    for j in range(9):
        x = (j + 1) * gap  # Position horizontale des lignes verticales
        pygame.draw.line(window, BLACK, (x, 0), (x, height), 2)  # Ligne verticale

    # Dessiner la frontière (rectangle autour de la grille)
    border_thickness = 1  # Épaisseur de la bordure
    pygame.draw.rect(window, BORDER_COLOR, (0, 0, width + 0.5, height + 0.5), border_thickness)

    # Mise à jour de l'écran
    pygame.display.flip()

# Quitter Pygame proprement
pygame.quit()
sys.exit()
