import React, { useState } from 'react';
import axios from "axios"
import './board.css';

// Initial piece setup (position is in 0-indexed row/column format)
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

  const getPiecePos = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("https://localhost:8000/get_piece_pos");
      console.log("Data submitted:", response.data);
    } catch (error) {
      console.error("Error getting position:", error);
    }
  }

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
      {initialPieces.map((piece) => (
        <div
          key={piece.id}
          className={`piece ${piece.color}`}
          style={{
            left: `${(piece.position.col / (cols - 1)) * 100}%`,
            top: `${(piece.position.row / (rows - 1)) * 100}%`,
          }}
        >
          {piece.type[0].toUpperCase()}
        </div>
      ))}
    </div>
  );
};

export default Board;
