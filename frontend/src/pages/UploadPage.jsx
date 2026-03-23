import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api/Client";
import "../App.css";

export default function UploadPage(props) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [fileError, setFileError] = useState("");
  const [sentenceCount, setSentenceCount] = useState(0);
  const [localTimer, setLocalTimer] = useState(0);
  const navigate = useNavigate();

// Effect to run the timer while 'loading' is true
  useEffect(() => {
      let interval;
      if (loading) {
        interval = setInterval(() => {
          setLocalTimer((prev) => prev + 1);
        }, 1000);
      } else {
        setLocalTimer(0); // Reset when not loading
      }
      return () => clearInterval(interval);
    }, [loading]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    const fileName = selectedFile.name.toLowerCase();
    const isAcceptedType = fileName.endsWith('.txt') || 
                           fileName.endsWith('.sent') || 
                           fileName.endsWith('.okpuncs');

    if (!isAcceptedType) {
      setFileError("⚠️ Invalid file type. Please select .txt or dataset files (.sent, .okpuncs).");
      setFile(null);
      return;
    }

    setFile(selectedFile);
    setFileError("");

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const count = text.split(/[.!?]+\s/).filter(s => s.trim().length > 0).length;
      setSentenceCount(count);

      if (count < 50) {
        setFileError(`⚠️ Narrative too short: ${count} sentences. Min 50 required.`);
      } else if (count > 200) {
        setFileError(`⚠️ Narrative too long: ${count} sentences. Max 200 allowed.`);
      }
    };
    reader.readAsText(selectedFile);
  };

  const handleUpload = async () => {
    if (!file || fileError) return;
    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setStatus("📤 Initiating Emotion Analysis Pipeline...");

    try {
      await api.post("/analyze/", formData);

      const interval = setInterval(async () => {
        const statusRes = await api.get("/status/");

        if (statusRes.data.status === "done") {
          clearInterval(interval);
          setStatus("✅ Analysis complete! Redirecting...");

          //Fetch results to get the OFFICIAL execution time from backend
          try {
            const resultRes = await api.get("/results/");
            if (resultRes.data.execution_time) {
              // Store it in the App.jsx state
              props.setExecutionTime(resultRes.data.execution_time);
            }
          } catch (err) {
            console.error("Could not fetch execution time", err);
          }

          setTimeout(() => {
            setLoading(false);
            navigate("/results");
          }, 1200);
        } else if (statusRes.data.status === "error") {
          clearInterval(interval);
          setStatus("❌ Analysis failed.");
          setLoading(false);
        }
      }, 2000);
    } catch (error) {
      console.error(error);
      setStatus("❌ Connection error.");
      setLoading(false);
    }
  };

  return (
    <div className="app-wrapper">
      <nav className="navbar">
        <div className="nav-logo-container" onClick={() => navigate("/")}>
          <div className="logo-icon">📈</div>
          <div className="logo-text-wrapper">
            <span className="logo-main-text">EMOi<span>-TRACK</span></span>
          </div>
        </div>
        <div className="nav-links">
          <span className="nav-item" onClick={() => navigate("/")}>Home</span>
          <span className="nav-item active">Upload</span>     
          <span className="nav-item" onClick={() => navigate("/results")}>Results</span>
          
          {/* MOVED: Researcher Name is now in the Navbar corner */}
          <div style={{ marginLeft: "20px", padding: "4px 12px", backgroundColor: "#eff6ff", borderRadius: "20px", fontSize: "12px", fontWeight: "700", color: "#3b82f6", display: "flex", alignItems: "center" }}>
            <span style={{ marginRight: "6px" }}>👤</span> {props.researcher || "Guest"}
          </div>
        </div>
      </nav>

      <main className="main-content">
        <div className="upload-card" style={{ textAlign: "center" }}>
          {/* CENTERED: Version Badge is now centered in the card */}
          <div style={{ display: "flex", justifyContent: "center", marginBottom: "20px" }}>
            <div style={{ padding: "6px 14px", backgroundColor: "#f1f5f9", borderRadius: "6px", fontSize: "11px", fontWeight: "800", color: "#475569", letterSpacing: "0.5px" }}>
              Negative Emotion RoBERTa / NEFI v1.0
            </div>
          </div>

          <h1>Narrative Analysis</h1>
          <p style={{ color: "#64748b", fontSize: "15px", marginBottom: "30px" }}>
            Welcome, <strong>{props.researcher}</strong>. Select a narrative to identify emotional shifts.
          </p>

          <div 
            className={`dropzone ${file ? "active-zone" : ""}`}
            onClick={() => document.getElementById("fileInput").click()}
            style={{ border: fileError ? "2px dashed #ef4444" : "2px dashed #e2e8f0", margin: "0 auto 20px" }}
          >
            <p style={{ fontWeight: "700", color: file ? "#0f172a" : "#94a3b8" }}>
              {file ? file.name : "Select .txt or dataset file"}
            </p>
            <input 
              id="fileInput" 
              type="file" 
              accept=".txt,.sent,.okpuncs" 
              hidden 
              onChange={handleFileChange} 
            />
          </div>

          {fileError && (
            <p style={{ color: "#ef4444", fontSize: "13px", fontWeight: "600", marginBottom: "15px" }}>
              {fileError}
            </p>
          )}

          <button className="primary-btn" onClick={handleUpload} disabled={!file || loading || !!fileError} style={{ width: "100%", maxWidth: "300px" }}>
            {loading ? `Analyzing...(${localTimer}s)` : "Analyze File"}
          </button>
          
          {status && <p style={{ fontSize: "13px", fontWeight: "600", marginTop: "15px" }}>{status}</p>}
        </div>
      </main>

      <footer style={{ padding: "40px", textAlign: "center", fontSize: "11px", color: "#94a3b8", fontWeight: "700", borderTop: "1px solid #e2e8f0" }}>
        Project Emoi-Track | {props.researcher || "Senuvi Jayasinghe"} | Computer Science
      </footer>
    </div>
  );
}