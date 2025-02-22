CREATE TABLE games (
    game_id SERIAL PRIMARY KEY,
    username VARCHAR(20) NOT NULL,
    board_state JSON NOT NULL DEFAULT  ('[[-1, -2, -3, -4, -5, -6, -7, -8, -9],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -10, 0, 0, 0, 0, 0, -11, 0],
        [-12, 0, -13, 0, -14, 0, -15, 0, -16],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [12, 0, 13, 0, 14, 0, 15, 0, 16],
        [0, 10, 0, 0, 0, 0, 0, 11, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9]]'),
    turn INT NOT NULL DEFAULT 1, 
    status VARCHAR(20) NOT NULL DEFAULT 'ongoing' -- Default game status
);