from fastapi import FastAPI, HTTPException
from database import get_db_connection
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend requests (Adjust if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)



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


@app.post("/create_user")
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

@app.post("/get_piece_pos")
def create_user(game_id: str):
    connection = get_db_connection()
    with connection.cursor() as cursor:

        cursor.execute("SELECT board_state FROM games WHERE game_id = %s", (game_id,))
        result = cursor.fetchone()

    connection.close()
    return {"message": "game created", "board": result["board_state"]}

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
