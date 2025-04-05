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


while True:
    draw_board(game.board)

    piece = int(input("Eneter a piece: "))

    legal_moves = piece_move.get_legal_moves(piece, game.board)

    print(f"Your legal moves are: {legal_moves}")

    choice = int(input("Eneter a choice: "))

    legal_move_chosen = legal_moves[choice]

    game.board = piece_move.make_move(piece, legal_move_chosen[0], legal_move_chosen[1], game.board)

    #马


    """class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()

        # Initial Convolutional Layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Policy Head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 10 * 9, action_size)

        # Value Head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(10 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, 10, 9)

        # Initial Convolutional Block
        x = F.relu(self.bn1(self.conv1(x)))

        # Policy Head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = F.softmax(self.policy_fc(policy), dim=1)

        # Value Head
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value"""