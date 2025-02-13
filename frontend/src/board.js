import React from "react";
import "./board.css"; // Create a CSS file for styling

const rows = 9;
const cols = 8;

const Board = () => {
  return (
    <div className="board-container">
        <div className="board">
        {Array.from({ length: rows * cols }).map((_, index) => {
            const x = index % cols;
            const y = Math.floor(index / cols);
            return <div key={index} className="cell">{x},{y}</div>;
        })}
        </div>
    </div>
  );
};

export default Board;
