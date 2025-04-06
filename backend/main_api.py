from fastapi import FastAPI, HTTPException
from database import get_db_connection
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import training_attributes
import piece_move
import numpy as np

app = FastAPI()

class BoardRequest(BaseModel):
    username: Optional[str] = None
    piece: Optional[int] = None
    board: Optional[List[List[int]]] = None
    turn: Optional[int] = None

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
def get_legal_moves(request: BoardRequest):
    print(request.piece, request.board)
    
    board_np = np.array(request.board, dtype=np.int32)
    legal_moves = piece_move.get_legal_moves(request.piece, board_np)

    return {"legal_moves": legal_moves}

@app.post("/save_board")
def save_board(request: BoardRequest):
    connection = get_db_connection()
    with connection.cursor() as cursor:
        board_json = json.dumps(request.board)
        cursor.execute("UPDATE games SET board_state = %s WHERE username = %s", (board_json, request.username))
        connection.commit()

    connection.close()
    return {"message": "board saved"}


@app.post("/flip_turn")
def flip_turn(request: BoardRequest):
    print(request.username)
    connection = get_db_connection()
    with connection.cursor() as cursor:
        turn =  1 - request.turn
        cursor.execute("UPDATE games SET turn = %s WHERE username = %s", (turn, request.username))
        connection.commit()

    connection.close()
    return {"new_turn": turn}


@app.post("/get_turn")
def get_turn(request: BoardRequest):
    connection = get_db_connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT turn FROM games WHERE username = %s", (request.username,))
        turn = cursor.fetchone()

    connection.close()
    return turn 

 
@app.post("/get_ai_moves")
def get_ai_moves(request: BoardRequest):

    AI_moves = training_attributes.generate_moves(request.board, request.turn)
    
    return {"AI_moves": AI_moves}


@app.post("/is_check")
def is_check(request: BoardRequest):
    board_1d = piece_move.encode_board_to_1d_board(request.board)
    is_check = piece_move.is_check(board_1d, request.turn)

    return {"is_check": is_check}

# uvicorn main_api:app --reload