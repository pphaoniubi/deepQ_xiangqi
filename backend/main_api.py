from fastapi import FastAPI, HTTPException
from database import get_db_connection

app = FastAPI()

# 📌 Get a user by ID
@app.get("/users/{user_id}")
def get_user(user_id: int):
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM appUser WHERE user_id = %s", (user_id,))
        user = cursor.fetchone()
    connection.close()
    
    if user:
        return {"user": user}
    else:
        raise HTTPException(status_code=404, detail="User not found")


@app.post("/create_users")
def create_user(name: str):
    connection = get_db_connection()
    with connection.cursor() as cursor:

        cursor.execute("SELECT COUNT(*) FROM appUser WHERE name = %s", (name,))
        result = cursor.fetchone()
        
        if result["COUNT(*)"] > 0:
            connection.close()
            raise HTTPException(status_code=400, detail="User with this name already exists")

        cursor.execute("INSERT INTO appUser (name) VALUES (%s)", (name,))
        connection.commit()
        new_user_id = cursor.lastrowid

    connection.close()
    return {"message": "User created", "user_id": new_user_id}

@app.post("/create_game")
def create_user(username: str):
    connection = get_db_connection()
    with connection.cursor() as cursor:

        cursor.execute("INSERT INTO games (username) VALUES (%s)", (username,))
        connection.commit()
        new_game_id = cursor.lastrowid

    connection.close()
    return {"message": "game created", "game_id": new_game_id}

# 📌 Get all games
@app.get("/games/")
def get_games():
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM games")
        games = cursor.fetchall()
    connection.close()
    return {"games": games}

# 📌 Get a game by ID
@app.get("/games/{game_id}")
def get_game(game_id: int):
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM games WHERE game_id = %s", (game_id,))
        game = cursor.fetchone()
    connection.close()
    
    if game:
        return {"game": game}
    else:
        raise HTTPException(status_code=404, detail="Game not found")

# 📌 Create a new game
@app.post("/games/")
def create_game(user_id: int, board_state: dict, turn: int, status: str):
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO games (user_id, board_state, turn, status) VALUES (%s, %s, %s, %s)",
            (user_id, str(board_state), turn, status)
        )
        connection.commit()
        new_game_id = cursor.lastrowid
    connection.close()
    return {"message": "Game created", "game_id": new_game_id}
