from game_state import game
import piece_move

def draw_board(board):
    
    print("   0   1   2   3   4   5   6   7   8")
    

    for y in range(10):
        row = f"{y} "
        for x in range(9):
            piece = board[y][x]
            if piece == 0:
                row += " .  "
            elif 0 < piece < 10:
                row += f'+{piece}  '
            elif piece >= 10:
                row += f'+{piece} '
            elif -10 < piece < 0:
                row += f'{piece}  '
            elif piece <= -10:
                row += f'{piece} '
        print(row)
    #马
# Exemple de configuration du plateau (Xiangqi)
# Utiliser 0 pour les cases vides et des symboles pour les pièces
game.board = [
    [-1, -2, -3, -4, -5, -6, -7, -8, -9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -10, 0, 0, 0, 0, 0, -11, 0],
    [-12, 0, -13, 0, -14, 0, -15, 0, -16],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [12, 0, 13, 0, 14, 0, 15, 0, 16],
    [0, 10, 0, 0, 0, 0, 0, 11, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
]

print(piece_move.encode_board_to_1d_board(game.board))

while True:
    draw_board(game.board)

    piece = int(input("Eneter a piece: "))

    legal_moves = piece_move.get_legal_moves(piece, game.board)

    print(f"Your legal moves are: {legal_moves}")

    choice = int(input("Eneter a choice: "))

    legal_move_chosen = legal_moves[choice]

    game.board = piece_move.make_move(piece, legal_move_chosen[0], legal_move_chosen[1], game.board)
