"""import numpy as np
from attributes import grid_x, grid_y

class game:
    piece_names = ["Black Chariot", "Black Horse", "Black Elephant", "Black Advisor", "Black General", "Black Cannon", "Black Soldier",
                   "Red Chariot", "Red Horse", "Red Elephant", "Red Advisor", "Red General", "Red Cannon", "Red Soldier"]
    
    def __init__(self):
        # List creation for piece position tracking
        self.piece_position = {
            name: [] for name in self.piece_names  # Initialize an empty list for each piece
        }
        self.piece_position["Black Chariot"].append((0,0))
        self.piece_position["Black Chariot"].append((0,8))
        self.piece_position["Black Horse"].append((0,1))
        self.piece_position["Black Horse"].append((0,7))
        self.piece_position["Black Elephant"].append((0,2))
        self.piece_position["Black Elephant"].append((0,6))
        self.piece_position["Black Advisor"].append((0,3))
        self.piece_position["Black Advisor"].append((0,5))
        self.piece_position["Black General"].append((0,4))
        self.piece_position["Black Cannon"].append((2,1))
        self.piece_position["Black Cannon"].append((2,7))
        self.piece_position["Black Soldier"].append((3,0))
        self.piece_position["Black Soldier"].append((3,2))
        self.piece_position["Black Soldier"].append((3,4))
        self.piece_position["Black Soldier"].append((3,6))
        self.piece_position["Black Soldier"].append((3,8))
        self.piece_position["Red Chariot"].append((9,0))
        self.piece_position["Red Chariot"].append((9,8))
        self.piece_position["Red Horse"].append((9,1))
        self.piece_position["Red Horse"].append((9,7))
        self.piece_position["Red Elephant"].append((9,2))
        self.piece_position["Red Elephant"].append((9,6))
        self.piece_position["Red Advisor"].append((9,3))
        self.piece_position["Red Advisor"].append((9,5))
        self.piece_position["Red General"].append((9,4))
        self.piece_position["Red Cannon"].append((7,1))
        self.piece_position["Red Cannon"].append((7,7))
        self.piece_position["Red Soldier"].append((6,0))
        self.piece_position["Red Soldier"].append((6,2))
        self.piece_position["Red Soldier"].append((6,4))
        self.piece_position["Red Soldier"].append((6,6))
        self.piece_position["Red Soldier"].append((6,8))

        # Board creation and pieces placement
        self.board = np.full((10, 9), None, dtype=object)
        self.board[0, 0] = "Black Chariot"
        self.board[0, 1] = "Black Horse"
        self.board[0, 2] = "Black Elephant"
        self.board[0, 3] = "Black Advisor"
        self.board[0, 4] = "Black General"
        self.board[0, 5] = "Black Advisor"
        self.board[0, 6] = "Black Elephant"
        self.board[0, 7] = "Black Horse"
        self.board[0, 8] = "Black Chariot"
        self.board[2, 1] = "Black Cannon"
        self.board[2, 7] = "Black Cannon"
        self.board[3, 0] = "Black Soldier"
        self.board[3, 2] = "Black Soldier"
        self.board[3, 4] = "Black Soldier"
        self.board[3, 6] = "Black Soldier"
        self.board[3, 8] = "Black Soldier"
        self.board[9, 0] = "Red Chariot"
        self.board[9, 1] = "Red Horse"
        self.board[9, 2] = "Red Elephant"
        self.board[9, 3] = "Red Advisor"
        self.board[9, 4] = "Red General"
        self.board[9, 5] = "Red Advisor"
        self.board[9, 6] = "Red Elephant"
        self.board[9, 7] = "Red Horse"
        self.board[9, 8] = "Red Chariot"
        self.board[7, 1] = "Red Cannon"
        self.board[7, 7] = "Red Cannon"
        self.board[6, 0] = "Red Soldier"
        self.board[6, 2] = "Red Soldier"
        self.board[6, 4] = "Red Soldier"
        self.board[6, 6] = "Red Soldier"
        self.board[6, 8] = "Red Soldier"

        # Dictionary for window to array coordinate-mapping
        self.coords_to_array_index = {}
        for index_x, window_y in enumerate(grid_y):
            for index_y, window_x in enumerate(grid_x):
                self.coords_to_array_index[(window_x, window_y)] = (index_x, index_y)


    def save_state(self, piece_name, x, y):
        new_pos = self.coords_to_array_index.get((x, y))
        # Update position
        past_pos = self.piece_position.get(piece_name)
        self.board[past_pos[0], past_pos[1]] = None
        self.board[new_pos[0], new_pos[1]] = piece_name
        self.piece_position[piece_name] = [new_pos[0], new_pos[1]]

    def get_position(self, piece_name):
        return self.piece_position.get(piece_name)
    
    def get_all_positions(self):
        return self.piece_position
    
    def get_board(self):
        return self.board

    def display_board(self):
        for row in self.board:
            print(row)

"""