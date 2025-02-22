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
  const { username } = useContext(UserContext);

  const getPiecePos = async () => {
    console.log("Fetching data for:", username);
    if (!username) return;

    try {
      const response = await axios.post(`http://localhost:8000/get_piece_pos?username=${username}`);
      setBoard(response.data.board);
    } catch (error) {
      console.error("Error getting position:", error);
    }
  };

  useEffect(() => {
    console.log("useEffect triggered. Username:", username); 
      getPiecePos();
  }, [username]);

  useEffect(() => {
  }, [board]); // Runs when `board` is updated

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

      {/* Render only the pieces from the board */}
      {Array.isArray(board) &&
        board.map((row, rowIndex) =>
          Array.isArray(row) &&
          row.map((piece, colIndex) => {
            if (piece === 0) return null; // Skip empty cells
            return (
              <div
                key={`${rowIndex}-${colIndex}`}
                className={`piece ${piece < 0 ? "black" : "red"}`}
                style={{
                  position: "absolute",
                  left: `${(colIndex / (cols - 1)) * 100}%`,
                  top: `${(rowIndex / (rows - 1)) * 100}%`,
                  transform: "translate(-50%, -50%)",
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
