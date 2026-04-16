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

  useEffect(() => {
    let interval;
    if (loading) {
      interval = setInterval(() => {
        setLocalTimer((prev) => prev + 1);
      }, 1000);
    } else {
      setLocalTimer(0);
    }
    return () => clearInterval(interval);
  }, [loading]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    const fileName = selectedFile.name.toLowerCase();
    const isAcceptedType = fileName.endsWith(".okpuncs");

    if (!isAcceptedType) {
      setFileError(
        "⚠️ Invalid file type. Please select a dataset files (.okpuncs).",
      );
      setFile(null);
      setSentenceCount(0);
      return;
    }

    setFile(selectedFile);
    setFileError("");

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const count = text
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0).length;

      setSentenceCount(count);

      if (count < 50) {
        setFileError(
          `⚠️ Narrative too short: ${count} sentences. Minimum 50 required.`,
        );
      } else if (count > 200) {
        setFileError(
          `⚠️ Narrative too long: ${count} sentences. Maximum 200 allowed.`,
        );
      } else {
        setFileError("");
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
          setStatus(
            `✅ ${statusRes.data.message || "Analysis complete! Redirecting..."}`,
          );

          try {
            const resultRes = await api.get("/results/");
            if (resultRes.data.execution_time) {
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
          setStatus(`❌ ${statusRes.data.message || "Analysis failed."}`);
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
            <span className="logo-main-text">
              EMOi<span>-TRACK</span>
            </span>
          </div>
        </div>

        <div className="nav-links">
          <span className="nav-item" onClick={() => navigate("/")}>
            Home
          </span>
          <span className="nav-item active">Upload</span>
          <span className="nav-item" onClick={() => navigate("/results")}>
            Results
          </span>

          <div className="researcher-badge">
            <span>👤</span> {props.researcher || "Guest"}
          </div>
        </div>
      </nav>

      <main className="main-content">
        <div className="upload-card">
          <div className="version-badge-wrapper">
            <div className="version-badge">
              Negative Emotion RoBERTa / NEFI v1.0
            </div>
          </div>

          <h1>Narrative Analysis</h1>

          <p className="upload-subtitle">
            Welcome, <strong>{props.researcher}</strong>. Select a narrative to
            identify emotional shifts.
          </p>

          <div
            className={`dropzone ${file ? "active-zone" : ""}`}
            onClick={() => document.getElementById("fileInput").click()}
          >
            <p className="dropzone-text">
              {file ? file.name : "Select a dataset file"}
            </p>

            <input
              id="fileInput"
              type="file"
              accept=".okpuncs"
              hidden
              onChange={handleFileChange}
            />
          </div>

          {file && !fileError && (
            <p className="file-info">
              Narrative length detected: {sentenceCount} sentences
            </p>
          )}

          {fileError && <p className="file-error">{fileError}</p>}

          <button
            className="primary-btn"
            onClick={handleUpload}
            disabled={!file || loading || !!fileError}
          >
            {loading ? `Analyzing...(${localTimer}s)` : "Analyze File"}
          </button>

          {status && <p className="upload-status">{status}</p>}

          {!loading && (
            <div className="method-card">
              <h3>Research Method Overview</h3>

              <div className="method-section">
                <p className="method-section-title">Backbone Model</p>
                <p className="method-section-text">
                  RoBERTa selected based on Macro-F1 performance and training
                  stability.
                </p>
              </div>

              <div className="method-section">
                <p className="method-section-title">Selective Prediction</p>
                <ul>
                  <li>Temperature (T): 2.5</li>
                  <li>Threshold (τ): 0.76</li>
                  <li>Low-confidence predictions → UNKNOWN</li>
                </ul>
              </div>

              <div className="method-section">
                <p className="method-section-title">
                  Narrative Analysis Outputs
                </p>
                <ul>
                  <li>Sentence-level emotion classification</li>
                  <li>Emotional trajectory modelling</li>
                  <li>NEFI instability scoring</li>
                  <li>Rupture point detection</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="footer">
        Project Emoi-Track | {"Senuvi Jayasinghe"} | Computer Science
        Undergraduate
      </footer>
    </div>
  );
}
