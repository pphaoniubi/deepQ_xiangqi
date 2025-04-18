import React, { useContext } from "react";
import { UserContext } from "./UserContext";
import { useNavigate } from "react-router-dom"; 
import axios from "axios"

const UsernamePrompt = () => {
  const { username, setUsername } = useContext(UserContext);
  const navigate = useNavigate(); 

  // Handle username submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log(username.trim())
    if (username.trim() !== "") {
        try {
            const response = await axios.post(`http://localhost:8000/create_game?username=${username}`);
            if (response.status === 200) { // Check if request is successful
                console.log("Response:", response.data);
                navigate("/game_board"); // Redirect to Dashboard after success
              }
          } catch (error) {
            console.error("Error creating user", error);
          }
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
        <form onSubmit={handleSubmit}>
          <h2>Choose your username</h2>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Enter username"
            required
          />
          <button type="submit">Submit</button>
        </form>
    </div>
  );
};

export default UsernamePrompt;
