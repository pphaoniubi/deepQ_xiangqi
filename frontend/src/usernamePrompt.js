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
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Top heading */}
      <div style={{ minHeight: "100vh", 
        display: "flex", 
        flexDirection: "column", 
        justifyContent: "center", 
        alignItems: "center",
         }}>
      <form onSubmit={handleSubmit} style={{ textAlign: "center" }}>
        <h1 style={{ marginBottom: "20px" }}>Welcome to MCTs AI</h1> {/* 👈 Now inside form */}
        <p style={{ marginBottom: "15px" }}>Choose your username</p>
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="Enter username"
          required
          style={{ padding: "10px", fontSize: "16px", marginBottom: "10px", width: "200px", marginRight: "7px" }}
        />
        
        <button type="submit" style={{ padding: "10px 20px", fontSize: "16px" }}>Submit</button>
      </form>
    </div>
    </div>
  );
};

export default UsernamePrompt;
