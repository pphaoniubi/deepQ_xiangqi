import React, { useState, useEffect, useContext, useCallback } from 'react';
import { UserContext } from "./UserContext";
import axios from "axios"
import './board.css';

// INCOMPLETE
const pieceMapping = {
  "-1": "車", "-2": "傌", "-3": "象", "-4": "士", "-5": "將", "-6": "士", "-7": "象", "-8": "傌", "-9": "車",
  "-10": "炮", "-11": "炮", "-12": "卒", "-13": "卒", "-14": "卒", "-15": "卒", "-16": "卒",
  "1": "車", "2": "傌", "3": "象", "4": "士", "5": "帥", "6": "士", "7": "象", "8": "傌", "9": "車",
  "10": "炮", "11": "炮", "12": "兵", "13": "兵", "14": "兵", "15": "兵", "16": "兵",
  "0": ""
};

const Board = () => {
  const cols = 9;
  const rows = 10;
  const [board, setBoard] = useState(Array(rows).fill().map(() => Array(cols).fill(0)));
  const [selectedPiece, setSelectedPiece] = useState(null);
  const [legalMoves, setLegalMoves] = useState([]);
  const { username } = useContext(UserContext);

  const getPiecePos = async () => {
    if (!username) return;

    try {
      const response = await axios.post(`http://localhost:8000/get_piece_pos?username=${username}`);
      setBoard(response.data.board);
    } catch (error) {
      console.error("Error getting position:", error);
    }
  };


  const handlePieceClick = async (rowIndex, colIndex, piece, board) => {
    if (piece === 0) return; // Ignore empty cells
  
    console.log("Selected Piece:", piece, "at", rowIndex, colIndex);
    setSelectedPiece({ row: rowIndex, col: colIndex, piece });
    setLegalMoves([]); // Reset previous legal moves
  
    try {
      const response = await axios.post(`http://localhost:8000/get_legal_moves`, {
        piece: piece,
        board: board
      });
  
      console.log("Legal Moves:", response.data.moves);
      setLegalMoves(response.data.moves);
    } catch (error) {
      console.error("Error fetching legal moves:", error);
    }
  };

  useEffect(() => {
      getPiecePos();
  }, [username]);

  return (
    <div className="xiangqi-board">
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

      {board.map((row, rowIndex) =>
        row.map((piece, colIndex) => {
          const isLegalMove = legalMoves.some(([r, c]) => r === rowIndex && c === colIndex);

          return (
            <div
              key={`${rowIndex}-${colIndex}`}
              className={`cell ${isLegalMove ? "legal-move" : ""}`}
              onClick={() =>
                piece !== 0
                  ? handlePieceClick(rowIndex, colIndex, piece)
                  : isLegalMove && handleCellClick(rowIndex, colIndex)
              }
              style={{
                position: "absolute",
                left: `${(colIndex / (cols - 1)) * 100}%`,
                top: `${(rowIndex / (rows - 1)) * 100}%`,
                transform: "translate(-50%, -50%)",
                backgroundColor: isLegalMove ? "rgba(0, 255, 0, 0.3)" : "transparent", 
              }}
            >
              {pieceMapping[piece.toString()]}
            </div>
          );
        })
      )}
    </div>
  );
};

export default Board;
