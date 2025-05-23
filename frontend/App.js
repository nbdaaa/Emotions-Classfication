// App.js
import React, { useState } from 'react';
import './App.css';
import logoImage from './logo.png'; // Cần thêm file logo vào project

function App() {
  const [text, setText] = useState('');
  const [emotion, setEmotion] = useState('');

  const handleAnalyze = () => {
    // Placeholder for AI prediction logic
    setEmotion('Happy'); // Replace with actual prediction result
  };

  return (
    <div className="app">
      <nav className="navbar">
        <img src={logoImage} alt="Logo" className="logo" />
      </nav>
      
      <header className="header">
        <h1>TEXT EMOTION CLASSIFICATION</h1>
      </header>
      
      <div className="content">
        <div className="input-section">
          <textarea
            placeholder="Enter your text here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button onClick={handleAnalyze}>Start analyzing</button>
        </div>
        
        <div className="output-section">
          <h2>Detected Emotion:</h2>
          <p>{emotion || "No emotion detected yet"}</p>
        </div>
      </div>
    </div>
  );
}

export default App;