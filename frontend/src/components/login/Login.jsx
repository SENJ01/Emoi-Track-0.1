import React, { useState } from "react";
import "./Login.css";

export default function Login({ setAuth }) {
  const [name, setName] = useState("");
  const [key, setKey] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (key === "uni2026") {
      setAuth(true, name);
    } else {
      alert("Invalid Access Key");
    }
  };

  return (
    <div className="landing-container">
      {/* LEFT SIDE: Technical Project Info */}
      <div className="landing-info">
        <div className="badge">PROJECT EMOi-TRACK | V1.0</div>
        
        <h1 className="landing-title">
          Decoding Emotional <br />
          <span>Trajectories</span> in Text.
        </h1>
        
        <p className="main-description">
          An advanced computational linguistics platform designed to map the 
          <strong> Narrative State-Space</strong> of qualitative data.
        </p>

        {/* THREE CORE PILLARS OF THE RESEARCH */}
        <div className="project-pillars">
          <div className="pillar">
            <div className="pillar-header">
              <span className="pillar-num">01</span>
              <h4>Emotion Detection</h4>
            </div>
            <p>Utilizes <strong>RoBERTa-base</strong> fine-tuned on custom emotion labels for high-granularity sentiment mapping across long-form text.</p>
          </div>
          
          <div className="pillar">
            <div className="pillar-header">
              <span className="pillar-num">02</span>
              <h4>Rupture Analysis</h4>
            </div>
            <p>Employs the <strong>NEFI (Negative Emotion Fracture Index)</strong> algorithm to identify sudden, significant emotional shifts or 'fractures'.</p>
          </div>
          
          <div className="pillar">
            <div className="pillar-header">
              <span className="pillar-num">03</span>
              <h4>Multi-Scale Logic</h4>
            </div>
            <p>Processes narratives at both <strong>Sentence</strong> and <strong>Segment</strong> levels to ensure temporal accuracy in trajectory plotting.</p>
          </div>
        </div>
        
        <div className="tech-stack-footer">
          <span>PyTorch</span>
          <span className="dot">•</span>
          <span>FastAPI</span>
          <span className="dot">•</span>
          <span>React.js</span>
        </div>
      </div>

      {/* RIGHT SIDE: Authentication Section */}
      <div className="auth-section">
        <div className="login-card">
          <div className="login-header">
            <h2>User Login</h2>
            <p>Please enter credentials to access the system.</p>
          </div>
          
          <form onSubmit={handleSubmit}>
            <div className="input-group">
              <label>User Name</label>
              <input 
                type="text" 
                placeholder="e.g. Dr.John" 
                required 
                onChange={(e) => setName(e.target.value)} 
              />
            </div>
            
            <div className="input-group">
              <label>Access Key</label>
              <input 
                type="password" 
                placeholder="••••••••" 
                required 
                onChange={(e) => setKey(e.target.value)} 
              />
            </div>
            
            <button type="submit" className="login-btn">
              Launch Analysis Workspace →
            </button>
          </form>
          
          <div className="login-footer">
            System Version: v1.0 | EMOi-TRACK Core Engine | Academic Edition
          </div>
        </div>
      </div>
    </div>
  );
}