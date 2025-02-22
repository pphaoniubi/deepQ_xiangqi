import React, { useState, useEffect, useContext } from 'react';
import { UserContext } from "./UserContext";
import axios from "axios"
import './board.css';

// INCOMPLETE
const piece_mapping = {
  "-1": "車", "-2": "傌", "-3": "象", "-4": "士", "-5": "將", "-6": "士", "-7": "象", "-8": "♜", "-9": "車",
  "-10": "♝", "-11": "♝", "-12": "♟", "-13": "♟", "-14": "♟", "-15": "♟", "-16": "♟",
  "1": "♖", "2": "♘", "3": "♗", "4": "♕", "5": "♔", "6": "♗", "7": "♘", "8": "♖", "9": "♖",
  "10": "♗", "11": "♗", "12": "♙", "13": "♙", "14": "♙", "15": "♙", "16": "♙",
  "0": ""
}
const initialPieces = [
  { id: 'r1', type: '車', color: 'black', position: { row: 0, col: 0 } },
  { id: 'r2', type: '車', color: 'black', position: { row: 0, col: 8 } },
  { id: 'h1', type: '傌', color: 'black', position: { row: 0, col: 1 } },
  { id: 'h2', type: '傌', color: 'black', position: { row: 0, col: 7 } },
  { id: 'r4', type: '象', color: 'black', position: { row: 0, col: 2 } },
  { id: 'r4', type: '象', color: 'black', position: { row: 0, col: 6 } },
  { id: 'r4', type: '士', color: 'black', position: { row: 0, col: 3 } },
  { id: 'r4', type: '士', color: 'black', position: { row: 0, col: 5 } },
  { id: 'r4', type: '將', color: 'black', position: { row: 0, col: 4 } },
  { id: 'r4', type: '炮', color: 'black', position: { row: 2, col: 1 } },
  { id: 'r4', type: '炮', color: 'black', position: { row: 2, col: 7 } },
  { id: 'r4', type: '卒', color: 'black', position: { row: 3, col: 0 } },
  { id: 'r4', type: '卒', color: 'black', position: { row: 3, col: 2 } },
  { id: 'r4', type: '卒', color: 'black', position: { row: 3, col: 4 } },
  { id: 'r4', type: '卒', color: 'black', position: { row: 3, col: 6 } },
  { id: 'r4', type: '卒', color: 'black', position: { row: 3, col: 8 } },

  { id: 'r1', type: '車', color: 'red', position: { row: 9, col: 0 } },
  { id: 'r2', type: '車', color: 'red', position: { row: 9, col: 8 } },
  { id: 'h1', type: '傌', color: 'red', position: { row: 9, col: 1 } },
  { id: 'h2', type: '傌', color: 'red', position: { row: 9, col: 7 } },
  { id: 'r4', type: '象', color: 'red', position: { row: 9, col: 2 } },
  { id: 'r4', type: '象', color: 'red', position: { row: 9, col: 6 } },
  { id: 'r4', type: '士', color: 'red', position: { row: 9, col: 3 } },
  { id: 'r4', type: '士', color: 'red', position: { row: 9, col: 5 } },
  { id: 'r4', type: '將', color: 'red', position: { row: 9, col: 4 } },
  { id: 'r4', type: '炮', color: 'red', position: { row: 7, col: 1 } },
  { id: 'r4', type: '炮', color: 'red', position: { row: 7, col: 7 } },
  { id: 'r4', type: '卒', color: 'red', position: { row: 6, col: 0 } },
  { id: 'r4', type: '卒', color: 'red', position: { row: 6, col: 2 } },
  { id: 'r4', type: '卒', color: 'red', position: { row: 6, col: 4 } },
  { id: 'r4', type: '卒', color: 'red', position: { row: 6, col: 6 } },
  { id: 'r4', type: '卒', color: 'red', position: { row: 6, col: 8 } },
];

const Board = () => {
  const boardSize = 450; // Total board size (adjust as needed)
  const cols = 9;
  const rows = 10;
  const cellSize = boardSize / (cols - 1); // Distance between crosses
  const [board, setBoard] = useState(Array(rows).fill().map(() => Array(cols).fill(null)));
  const { username } = useContext(UserContext);

  const getPiecePos = async (e) => {

    try {
      const response = await axios.post(`http://localhost:8000/get_piece_pos?game_id=${username}`);
      setBoard(response.data)
      console.log("Data submitted:", response.data);
    } catch (error) {
      console.error("Error getting position:", error);
    }
  }

  useEffect(() => {

      getPiecePos();
    
  }, [username]);

  return (
    <div className="xiangqi-board">
      {/* Board Background */}
      <div className="board-lines">
        {/* Horizontal Lines */}
        {Array.from({ length: rows }).map((_, i) => (
          <div
            key={`h-${i}`}
            className="line horizontal"
            style={{ top: `${(i / (rows - 1)) * 100}%` }}
          />
        ))}
        {/* Vertical Lines */}
        {Array.from({ length: cols }).map((_, i) => (
          <div
            key={`v-${i}`}
            className="line vertical"
            style={{ left: `${(i / (cols - 1)) * 100}%` }}
          />
        ))}
      </div>

      {/* Pieces */}
      {/*initialPieces.map((piece) => (
        <div
          key={piece.id}
          className={`piece ${piece.color}`}
          style={{
            left: `${(piece.position.col / (cols - 1)) * 100}%`,
            top: `${(piece.position.row / (rows - 1)) * 100}%`,
          }}
        >
          {piece.type[0].toUpperCase()}
        </div >
      ))*/}

      {board.map((row, rowIndex) => (
        <div key={rowIndex} style={{ display: "flex" }}>
          {row.map((cell, colIndex) => (
            <div 
              key={colIndex} 
              style={{ padding: "10px", border: "1px solid black" }}
            >
              {cell}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

export default Board;
