class game:
    piece_name = ""
    init_x = 0
    init_y = 0
    new_x = 0
    new_y = 0
    board = [
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

    def save_state(self, piece_name, init_x, init_y, new_x, new_y):
        self.piece_name = piece_name
        self.init_x = init_x
        self.init_y = init_y
        self.new_x = new_x
        self.new_y = new_y