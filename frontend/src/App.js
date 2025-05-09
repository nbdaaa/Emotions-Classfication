// App.js
import React, { useState } from "react";
import "./App.css";
import RiveAnimation from "./RiveAnimation";
import Swal from "sweetalert2";
import logoImage from "./logo.png"; // Cần thêm file logo vào project

function App() {
    const [text, setText] = useState("");
    const [riveEmotion, setRiveEmotion] = useState("Idle");
    const [emotionClassify, setEmotionClassify] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [history, setHistory] = useState([]);

    const handleAnalyze = async () => {
        if (!text.trim()) {
            Swal.fire({
                title: "Oops...",
                text: "Please enter some text to analyze!",
                icon: "warning",
                confirmButtonColor: "#007bff",
                background: "#fff",
                customClass: {
                    popup: "animated fadeInDown",
                },
            });
            return;
        }

        setIsLoading(true);
        try {
            // const response = await fetch("http://localhost:5000/predict", {
            //     method: "POST",
            //     headers: {
            //         "Content-Type": "application/json",
            //     },
            //     body: JSON.stringify({ text: text }),
            // });

            // const data = await response.json();

            console.log("Change emotions");
            const emotions = ["Joy", "Sadness", "Anger", "Fear", "Love", "Suprise"];
            const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)];
            // Chuyển đổi cảm xúc từ backend sang tên animation tương ứng
            const emotionMap = {
                Joy: "happy",
                Sadness: "sad",
                Anger: "Angry",
                Fear: "stressed",
                Suprise: "excited",
                Love: "whistling",
            };

            const newEmotion = emotionMap[randomEmotion];
            setEmotionClassify(randomEmotion);
            setRiveEmotion(newEmotion);

            // Add to history
            setHistory((prev) =>
                [
                    {
                        text: text,
                        emotion: randomEmotion,
                        timestamp: new Date().toLocaleTimeString(),
                    },
                    ...prev,
                ].slice(0, 10)
            ); // Keep last 10 items
        } catch (error) {
            console.error("Error:", error);
            setRiveEmotion("Idle");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="app">
            <div className="app-title">Text Emotion Classification</div>

            <div className="history-section">
                <h2>History</h2>
                {history.map((item, index) => (
                    <div key={index} className="history-item">
                        <p>
                            <strong>Text:</strong> {item.text}
                        </p>
                        <p>
                            <strong>Emotion:</strong> {item.emotion}
                        </p>
                        <p>
                            <small>{item.timestamp}</small>
                        </p>
                    </div>
                ))}
            </div>

            <div className="main-content">
                <div className="input-section">
                    <textarea placeholder="Enter your text here..." value={text} onChange={(e) => setText(e.target.value)} />
                    <button onClick={handleAnalyze} disabled={isLoading}>
                        {isLoading ? "Analyzing..." : "Start analyzing"}
                    </button>
                </div>

                <div className="animation-container">
                    <div className="animation-section">
                        <RiveAnimation emotion={riveEmotion} />
                    </div>
                    <div className="emotion-display">
                        <h2>Detected Emotion</h2>
                        <p>{emotionClassify}</p>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
