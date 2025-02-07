class game:
    piece_name = ""
    init_x = 0
    init_y = 0
    new_x = 0
    new_y = 0
    board = [[0] * 9 for _ in range(10)]

    def save_state(self, piece_name, init_x, init_y, new_x, new_y):
        self.piece_name = piece_name
        self.init_x = init_x
        self.init_y = init_y
        self.new_x = new_x
        self.new_y = new_y