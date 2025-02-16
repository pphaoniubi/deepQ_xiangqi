CREATE TABLE games (
    game_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    board_state JSONB NOT NULL, 
    turn INT NOT NULL,
    status      
);


create table appUser (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(255)
)