import React, { useState, useEffect, useContext } from 'react';
import { UserContext } from "./UserContext";
import axios from "axios"
import './board.css';

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
  const [turn, setTurn] = useState(null);
  const [gameOver, setGameOver] = useState(false);
  const { username } = useContext(UserContext);


  const getBoard = async () => {
    if (!username || gameOver) return;

    try {
      const response = await axios.post(`http://localhost:8000/get_piece_pos?username=${username}`);
      
      const blackGeneralExists = response.data.board?.some(row => row.includes(-5));
      const redGeneralExists = response.data.board?.some(row => row.includes(5));

      if (!blackGeneralExists) {
        console.log("red wins!!");
      }
      else if (!redGeneralExists) {
        console.log("black wins!!");
      }

      setBoard(response.data.board);
    } catch (error) {
      console.error("Error getting position:", error);
    }
  };

  
  const isWinner = (board) => {
    if (!username) return;

    try {
      
      const blackGeneralExists = board?.some(row => row.includes(-5));
      const redGeneralExists = board?.some(row => row.includes(5));

      if (!blackGeneralExists) {
        console.log("red wins!!");
        setGameOver(true);
      }
      else if (!redGeneralExists) {
        console.log("black wins!!");
        setGameOver(true);
      }

    } catch (error) {
      console.error("Error checking winner:", error);
    }
  };

  const handlePieceClick = async (rowIndex, colIndex, piece, board) => {
    if (piece === 0 || gameOver) return;

    const turnResponse = await axios.post(`http://localhost:8000/get_turn`, {
      username: username
    }, 
    { headers: { "Content-Type": "application/json" }}
    );

    if ((turnResponse.data.turn === 1 && piece < 0) || (turnResponse.data.turn === 0 && piece > 0)) 
      { console.log("wrong turn!");
        return;
      }
    setTurn(turnResponse.data.turn);
  
    setSelectedPiece({ row: rowIndex, col: colIndex, piece });
    setLegalMoves([]);

    console.log("Selected Piece:", piece, "at", rowIndex, colIndex);

    try {
      const response = await axios.post(`http://localhost:8000/get_legal_moves`, {
        piece: piece,
        board: board
      }, 
      { headers: { "Content-Type": "application/json" }}
    );
    const reversedMoves = response.data.legal_moves.map(move => move.reverse());
      console.log("Legal Moves:", reversedMoves);
      setLegalMoves(reversedMoves);
    } catch (error) {
      console.error("Error fetching legal moves:", error);
    }
  };

  const cancelPieceClick = (piece) => {
    if (selectedPiece)
      if(piece === selectedPiece.piece)
      {
        console.log("selectedPiece.piece", selectedPiece.piece)
        setSelectedPiece(null);
        setLegalMoves([]);
        return;
      }
  }

  const handleMoveClick = async (rowIndex, colIndex) => {
    if (!selectedPiece || !legalMoves || gameOver) return;

    const isLegal = legalMoves.some(
      (move) => move[0] === rowIndex && move[1] === colIndex
    );
  
    if (!isLegal) return;
    
    const newBoard = board.map((row) => [...row]);
    newBoard[rowIndex][colIndex] = selectedPiece.piece;
    newBoard[selectedPiece.row][selectedPiece.col] = 0;

    setBoard(newBoard);
    
    await axios.post(`http://localhost:8000/save_board`, {
      username: username,
      board: newBoard
    }, 
    { headers: { "Content-Type": "application/json" }}
  );

    await axios.post(`http://localhost:8000/flip_turn`, {     // flip turn
      username: username,
      turn: turn
    }, 
    { headers: { "Content-Type": "application/json" }}
  );

    setSelectedPiece(null);
    setLegalMoves([]);

    isWinner(newBoard);
  };

  useEffect(() => {
    getBoard();
  }, [username]);

  return (
    <div className="xiangqi-board">
      <div className="board-lines">
        {gameOver && <div className="game-over">Game Over!</div>}
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

      {Array.isArray(board) &&
        board.map((row, rowIndex) =>

          Array.isArray(row) &&
          row.map((piece, colIndex) => {

            const isLegalMove = legalMoves.some(([r, c]) => r === rowIndex && c === colIndex);

            return (
              <div
                key={`${rowIndex}-${colIndex}`}
                className={`
                ${piece < 0 ? "piece black" : ""} 

                ${piece > 0 ? "piece red" : ""} 

                ${isLegalMove ? "legal-move" : ""}`}

                onClick={() => {
                  if (gameOver) return;
                  
                  if (piece !== 0) {
                    if (!selectedPiece)
                      handlePieceClick(rowIndex, colIndex, piece, board);

                    else if(selectedPiece && piece === selectedPiece.piece) // cancel
                      cancelPieceClick(piece);

                    else if(selectedPiece && piece !== selectedPiece.piece
                      && Math.sign(piece) === Math.sign(selectedPiece.piece)
                    ) // reselect
                    {
                      cancelPieceClick(piece);
                      handlePieceClick(rowIndex, colIndex, piece, board);
                    }

                    else if (selectedPiece && Math.sign(piece) !== Math.sign(selectedPiece.piece))
                      handleMoveClick(rowIndex, colIndex);
                  } else {
                    if (isLegalMove) {
                      handleMoveClick(rowIndex, colIndex);
                    }
                  }
                }}

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
