import './App.css';
import Board from './board';
import UsernamePrompt from './usernamePrompt';
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import { UserProvider } from "./UserContext";

function App() {
  return (
    <UserProvider>
      <Router>
        <nav>
        </nav>
          <Routes>
            <Route path="/" element={<UsernamePrompt />} />
            <Route path="/game_board" element={<Board />} />
          </Routes>
        </Router>
    </UserProvider>
  );
}

export default App;
