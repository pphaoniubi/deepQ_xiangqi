from fastapi import FastAPI, HTTPException
from database import get_db_connection
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import AI.board_piece

app = FastAPI()

class LegalMovesRequest(BaseModel):
    piece: int
    board: List[List[int]]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
def create_user(username: str):
    connection = get_db_connection()
    with connection.cursor() as cursor:

        cursor.execute("SELECT board_state FROM games WHERE username = %s", (username,))
        result = cursor.fetchone()

    connection.close()

    board_data = json.loads(result["board_state"])
    return {"message": "board fetched", "board": board_data}

@app.post("/get_legal_moves")
def get_legal_moves(request: LegalMovesRequest):
    print(request.piece, request.board)
    legal_moves = AI.board_piece.get_legal_moves(request.piece, request.board)

    return {"legal_moves": legal_moves}