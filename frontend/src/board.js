import React, { useState, useEffect, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
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
  const [turn, setTurn] = useState(1);
  const [gameOver, setGameOver] = useState(false);
  const { username } = useContext(UserContext);
  const [playerMoved, setPlayerMoved] = useState(false);
  const [isCheckmate, setIsCheckmate] = useState(false);
  const [userSteps, setUserSteps] = useState(0);
  const [botSteps, setBotSteps] = useState(0);

  const navigate = useNavigate();


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
      console.log(response.data.board);
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

  
  const AI_Turn = async () => {
    if (!username || gameOver) return;

    const turnResponse = await axios.post(`http://localhost:8000/get_turn`, {
        username: username
    }, { headers: { "Content-Type": "application/json" } });

    setTurn(turnResponse.data.turn);
    console.log("AI turn", turnResponse.data.turn);
    if (turnResponse.data.turn === 1) return; // If it's not AI's turn, exit

    try {
        const response = await axios.post(`http://localhost:8000/get_ai_moves`, {
            board: board, 
            turn: turn 
        }, { headers: { "Content-Type": "application/json" } });

        const [piece, [destRow, destCol]] = response.data.AI_moves;

        // Find the original position of the piece in the board
        let originalRow = -1;
        let originalCol = -1;

        for (let r = 0; r < board.length; r++) {
            for (let c = 0; c < board[r].length; c++) {
                if (board[r][c] === piece) {  // Found the piece in the board
                    originalRow = r;
                    originalCol = c;
                    break;
                }
            }
        }

        setBotSteps(prev => prev + 1);

        if (originalRow === -1 || originalCol === -1) {
            console.error("Error: AI piece not found on the board!");
            return;
        }

        console.log(`AI Moving ${piece} from (${originalRow}, ${originalCol}) to (${destRow}, ${destCol})`);

        //  Update board with AI move
        const newBoard = board.map(row => [...row]); // Copy board
        newBoard[destRow][destCol] = piece;  // Move piece to destination
        newBoard[originalRow][originalCol] = 0;  // Clear original position

        setBoard(newBoard);

        // Save updated board
        await axios.post(`http://localhost:8000/save_board`, {
            username: username,
            board: newBoard
        }, { headers: { "Content-Type": "application/json" } });

        // Flip turn
        const response_new_turn = await axios.post(`http://localhost:8000/flip_turn`, {
            username: username,
            turn: turn
        }, { headers: { "Content-Type": "application/json" } });

        setTurn(response_new_turn.data.new_turn);

        getBoard();
        setPlayerMoved(false);
        setSelectedPiece(null);
        isWinner(newBoard);

    } catch (error) {
        console.error("Error getting AI position:", error);
    }
};

  const handlePieceClick = async (rowIndex, colIndex, piece, board) => {
    if (piece <= 0 || gameOver) return;

    const turnResponse = await axios.post(`http://localhost:8000/get_turn`, {
      username: username
    }, 
    { headers: { "Content-Type": "application/json" }}
    );

    if (turnResponse.data.turn === -1) {
      console.log("AI's turn!");
      // console.log("Checking AI_Turn: turn =", turn, ", playerMoved =", playerMoved);
      return;
    }

    setTurn(turnResponse.data.turn);
    console.log("checking turn ", turnResponse.data.turn);
  
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
        setSelectedPiece(null);
        setLegalMoves([]);
        return;
      }
  }

  const handleMoveClick = async (rowIndex, colIndex) => {
    if (!selectedPiece || !legalMoves || gameOver) return;

    console.log("inside")
    const isLegal = legalMoves.some(
      (move) => move[0] === rowIndex && move[1] === colIndex
    );
  
    if (!isLegal) return;
    
    const newBoard = board.map((row) => [...row]);
    newBoard[rowIndex][colIndex] = selectedPiece.piece;
    newBoard[selectedPiece.row][selectedPiece.col] = 0;

    setBoard(newBoard);
    setUserSteps(prev => prev + 1);
    await axios.post(`http://localhost:8000/save_board`, {
      username: username,
      board: newBoard
    }, 
    { headers: { "Content-Type": "application/json" }}
  );

    const response_new_turn = await axios.post(`http://localhost:8000/flip_turn`, {     // tuen: 1 is red, -1 is black 
      username: username,
      turn: turn
    }, 
    { headers: { "Content-Type": "application/json" }}
    );

    setTurn(response_new_turn.data.new_turn);
    setSelectedPiece(null);
    setLegalMoves([]);
    setPlayerMoved(true);
    isWinner(newBoard);
  };


  const is_check = async (board, turn) => {
    const response_checkmate = await axios.post(`http://localhost:8000/is_check`, {
      board: board,
      turn: turn
    }, 
    { headers: { "Content-Type": "application/json" }}
  );
  setIsCheckmate(response_checkmate.data.is_check)
  }

  useEffect(() => {
    getBoard();
  }, [username, turn]);


  useEffect(() => {
    {is_check(board, turn)}
  }, [board, turn]);

  useEffect(() => {
    console.log("Checking AI_Turn: turn =", turn, ", playerMoved =", playerMoved);
    
    if (turn === -1  && playerMoved) {
        console.log("AI_Turn is being called!");
        AI_Turn();
    }
}, [turn, playerMoved]);

  return (
    <div className="xiangqi-container">
      {gameOver && (
      <div className="popup-overlay">
        <div className="popup-content">
          <h2>{turn === 1 ? 'You Lost!' : 'You Won!'}</h2>
          <button onClick={() => navigate('/')}>Start Over</button>
        </div>
      </div>
    )}

      <div className={`user-box left ${turn === 1 ? 'active-turn' : ''}`}>
        User: {username} <br /><br />
        Steps: {userSteps}  
      </div>
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
      <div className={`user-box right ${turn === -1 ? 'active-turn' : ''}`}>
        Bot <br /><br />
        Steps: {botSteps}
      </div>
    </div>
  );
};

export default Board;
