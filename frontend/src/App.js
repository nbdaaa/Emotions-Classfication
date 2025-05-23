import React, { useState } from "react";
import "./App.css";
import RiveAnimation from "./RiveAnimation";

function App() {
    const [sentence, setSentence] = useState("");
    const [error, setError] = useState("");
    const [selectedModel, setSelectedModel] = useState("logistic");
    const [riveEmotion, setRiveEmotion] = useState("Idle");
    const [emotionClassify, setEmotionClassify] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [history, setHistory] = useState([]);

    const models = [
        { id: "logistic", name: "Logistic Regression" },
        { id: "decision_tree", name: "Decision Tree" },
        { id: "linear_svc", name: "Linear SVC" },
        { id: "multinomial_nb", name: "Multinomial Naive Bayes" },
        { id: "transformer", name: "Transformer" },
        { id: "transformer_rope", name: "Transformer with ROPE" },
    ];

    // Chuyển handleAnalyze thành hàm gửi request
    const handleAnalyze = async () => {
        if (!sentence.trim()) {
            setError("Please enter some text to analyze.");
            return;
        }

        setIsLoading(true);
        setError("");
        setRiveEmotion("Idle");
        setEmotionClassify("");

        try {
            console.log("Sending request to backend...");
            const response = await fetch("http://localhost:5000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sentence, model: selectedModel }),
            });

            const data = await response.json();
            console.log("Received response from backend:", data);
            if (!response.ok) {
                throw new Error(data.error || "Something went wrong");
            }

            setRiveEmotion(data.emotion);
            setEmotionClassify(data.emotion);

            setHistory((prev) =>
                [
                    {
                        text: sentence,
                        emotion: data.emotion,
                        timestamp: new Date().toLocaleTimeString(),
                    },
                    ...prev,
                ].slice(0, 10)
            );
        } catch (err) {
            setError(err.message);
            setRiveEmotion("Idle");
        } finally {
            setIsLoading(false);
        }
    };

    // Bỏ handleSubmit vì form chỉ chứa select, không submit gì

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
                    <textarea placeholder="Enter your text here..." value={sentence} onChange={(e) => setSentence(e.target.value)} />
                    <div className="model-selection">
                        <button className="analyze-button" onClick={handleAnalyze} disabled={isLoading}>
                            {isLoading ? "Analyzing..." : "Start analyzing"}
                        </button>
                        <form className="input-form" onSubmit={(e) => e.preventDefault()}>
                            <div className="model-selector">
                                <label htmlFor="model">Select Model:</label>
                                <select id="model" value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} className="model-select">
                                    {models.map((model) => (
                                        <option key={model.id} value={model.id}>
                                            {model.name}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </form>
                    </div>
                    {error && <p className="error-message">{error}</p>}
                </div>

                <div className="animation-container">
                    <div className="animation-section">
                        <RiveAnimation emotion={riveEmotion} />
                    </div>
                    <div className="emotion-display">
                        <h2>Detected Emotion</h2>
                        <p>{emotionClassify || "N/A"}</p>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
