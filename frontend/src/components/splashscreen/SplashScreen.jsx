import React, { useEffect, useState } from "react";
import "./SplashScreen.css";

const SplashScreen = ({ onFinished }) => {
  const [isFadingOut, setIsFadingOut] = useState(false);

  useEffect(() => {
    // 1. Stay on screen for 2.5 seconds
    const timer = setTimeout(() => {
      setIsFadingOut(true); // Start the fade-out animation
      
      // 2. Wait for the 0.5s CSS transition to finish before switching pages
      setTimeout(onFinished, 300); 
    }, 800);

    return () => clearTimeout(timer);
  }, [onFinished]);

  return (
    <div className={`splash-overlay ${isFadingOut ? "exit" : ""}`}>
      <div className="splash-content">
        <div className="splash-logo">📈</div>
        <h1 className="brand-name">EMOi<span>-TRACK</span></h1>
        <p className="tagline">Narrative Intelligence System</p>
        
        <div className="loader-track">
          <div className="loader-fill"></div>
        </div>
      </div>
    </div>
  );
};

export default SplashScreen;