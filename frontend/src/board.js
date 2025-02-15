import React from 'react';
import './board.css';

// Initial piece setup (position is in 0-indexed row/column format)
const initialPieces = [
  { id: 'r1', type: 'rook', color: 'red', position: { row: 0, col: 0 } },
  { id: 'r2', type: 'rook', color: 'red', position: { row: 0, col: 8 } },
  { id: 'h1', type: 'horse', color: 'red', position: { row: 0, col: 1 } },
  { id: 'r3', type: 'rook', color: 'black', position: { row: 9, col: 0 } },
  { id: 'r4', type: 'rook', color: 'black', position: { row: 9, col: 8 } },
  // Add all other pieces...
];

const Board = () => {
  const boardSize = 450; // Total board size (adjust as needed)
  const cols = 9;
  const rows = 10;
  const cellSize = boardSize / (cols - 1); // Distance between crosses

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
