import React, { useEffect, useRef } from "react";
import { Rive } from "@rive-app/canvas";

function RiveAnimation({ emotion }) {
    const canvasRef = useRef(null);
    const riveRef = useRef(null);
    const dpr = window.devicePixelRatio || 1;

    useEffect(() => {
        async function loadRive() {
            try {
                const res = await fetch("/snowwee.riv");
                const buffer = await res.arrayBuffer();

                // Set canvas pixel-perfect
                const canvas = canvasRef.current;
                canvas.width = 500 * dpr;
                canvas.height = 500 * dpr;
                canvas.style.width = "500px";
                canvas.style.height = "500px";

                const rive = new Rive({
                    buffer: buffer,
                    canvas: canvas,
                    autoplay: true,
                    onLoad: () => {
                        console.log("Rive animation loaded");
                        riveRef.current = rive;
                        console.log("State machine:", rive.stateMachineNames);
                        console.log("Animation names:", rive.animationNames);
                        rive.play("Idle");
                    },
                });
            } catch (error) {
                console.error("Error loading Rive animation:", error);
            }
        }

        loadRive();
    }, [dpr]);

    // Xử lý khi emotion thay đổi
    useEffect(() => {
        if (riveRef.current && emotion) {
            const validEmotions = ["happy", "sad", "excited", "Angry", "Idle", "whistling", "stressed"];
            if (validEmotions.includes(emotion)) {
                console.log("Playing emotion:", emotion);
                riveRef.current.stop();
                riveRef.current.play("Idle");
                riveRef.current.play(emotion);

                const interval = setInterval(() => {
                    riveRef.current.stop();
                    riveRef.current.play(emotion);
                }, 3000);

                return () => clearInterval(interval);
            }
        }
    }, [emotion]);

    return (
        <canvas
            ref={canvasRef}
            style={{
                objectFit: "contain",
                imageRendering: "pixelated",
                border: "0",
            }}
        />
    );
}

export default RiveAnimation;
