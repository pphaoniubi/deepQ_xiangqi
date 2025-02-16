from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import SessionLocal, Piece

app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def home():
    return {"message": "Xiangqi Piece Position API"}

@app.get("/positions")
def get_piece_positions(db: Session = Depends(get_db)):
    pieces = db.query(Piece).all()
    piece_dict = {}

    for piece in pieces:
        if piece.color not in piece_dict:
            piece_dict[piece.color] = {}
        piece_dict[piece.color][piece.name] = {"position": piece.position}

    return {"positions": piece_dict}
