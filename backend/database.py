from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# MySQL Database URL (Change username, password, and database_name)
DATABASE_URL = "mysql+pymysql://pphaoniubi:12345678pP!@localhost/xiangqi_db"

# Create Engine
engine = create_engine(DATABASE_URL)

# Create Session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base Model
Base = declarative_base()

# Define Table Model for Piece Positions
class Piece(Base):
    __tablename__ = "piece_positions"
    id = Column(String, primary_key=True, index=True)
    color = Column(String, index=True)  # "red" or "black"
    name = Column(String, index=True)   # Piece name (e.g., "chariot1")
    position = Column(String)           # Position on board (e.g., "A1")
