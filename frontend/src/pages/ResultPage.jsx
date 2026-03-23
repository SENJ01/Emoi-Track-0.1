import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api/Client";
import ResultDashboard from "../components/dashboard/ResultDashboard";
import "../App.css";

export default function ResultPage(props) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    async function fetchResults() {
      try {
        const res = await api.get("/results/");
        setData(res.data);
      } catch (error) {
        console.error("Error fetching results:", error);
      } finally {
        setLoading(false);
      }
    }
    fetchResults();
  }, []);

  const exportToCSV = () => {
    if (!data || !data.segments) return;
    const headers = ["Segment_Index", "Predicted_Emotion", "NEFI_Score", "Is_Rupture", "Text_Content"];
    const metaInfo = [["Researcher", props.researcher || "Anonymous"], ["Project", "EMOi-TRACK"], ["Date", new Date().toLocaleDateString()]];
    
    const rows = data.segments.map((seg, index) => [
      index + 1,
      seg.emotion,
      seg.nefi_score?.toFixed(4) || "0.0000",
      seg.is_rupture ? "YES" : "NO",
      `"${seg.text.replace(/"/g, '""')}"` 
    ]);

    // Join meta, empty spacer, headers, and data
    const csvContent = [
        ...metaInfo, 
        [], 
        headers, 
        ...rows
    ].map(e => e.join(",")).join("\n");

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `EmoiTrack_${props.researcher || "Analysis"}_Export.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (loading) {
    return (
      <div className="app-wrapper">
        <div className="main-content">
          <p style={{ fontWeight: "700", color: "#64748b" }}>📥 Fetching Analysis Results...</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="app-wrapper">
        <div className="main-content">
          <div className="upload-card">
            <p style={{ fontWeight: "700" }}>No results found.</p>
            <button className="primary-btn" onClick={() => navigate("/upload")} style={{ marginTop: "20px" }}>
              Go to Upload
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-wrapper">
      <nav className="navbar">
        <div className="nav-logo-container" onClick={() => navigate("/")} style={{ cursor: "pointer" }}>
          <div className="logo-icon">📈</div>
          <div className="logo-text-wrapper">
            <span className="logo-main-text">EMOi<span>-TRACK</span></span>
          </div>
        </div>
        <div className="nav-links">
          <span className="nav-item" onClick={() => navigate("/")}>Home</span>
          <span className="nav-item" onClick={() => navigate("/upload")}>Upload</span>     
          <span className="nav-item active">Results</span>
          
          {/*Add the same Researcher Badge from UploadPage */}
          <div style={{ marginLeft: "20px", padding: "4px 12px", backgroundColor: "#eff6ff", borderRadius: "20px", fontSize: "12px", fontWeight: "700", color: "#3b82f6", display: "flex", alignItems: "center" }}>
            <span style={{ marginRight: "6px" }}>👤</span> {props.researcher || "Guest"}
          </div>
        </div>
      </nav>

      <main className="main-content" style={{ display: "block", paddingTop: "40px" }}>
        <div className="results-container" style={{ maxWidth: "1200px", margin: "0 auto", padding: "0 40px" }}>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '30px' }}>
            <header style={{ textAlign: 'left' }}>
              <div className="system-status-pill">
                <span className="pulse-icon"></span>
                System Online | Latency: <strong>{props.executionTime || "0.00"}s</strong>
              </div>
              <h1 style={{ fontWeight: "900", fontSize: "2.5rem", margin: "0", letterSpacing: "-1.5px" }}>
                Analysis Dashboard
              </h1>
              <p style={{ color: "#64748b", fontWeight: "600", marginTop: "5px" }}>
                Showing results for researcher: <span style={{ color: "#3b82f6" }}>{props.researcher}</span>
              </p>
            </header>

            <div style={{ display: 'flex', gap: '10px' }}>
              <button 
                onClick={exportToCSV}
                className="csv-export-btn"
                style={{ backgroundColor: '#10b981', color: 'white', padding: '10px 18px', borderRadius: '8px', border: 'none', fontWeight: '800', fontSize: '12px', cursor: 'pointer', textTransform: 'uppercase' }}
              >
                📊 Export CSV
              </button>
              <button 
                className="primary-btn" 
                style={{ padding: '10px 18px', width: 'auto', fontSize: '12px' }}
                onClick={() => window.print()} 
              >
                📄 PDF Report
              </button>
            </div>
          </div>

          <ResultDashboard data={data} researcher={props.researcher} />
          
        </div>
      </main>

      <footer style={{ padding: "40px", textAlign: "center", fontSize: "11px", color: "#94a3b8", fontWeight: "700", borderTop: "1px solid #e2e8f0", backgroundColor: "white", marginTop: "40px" }}>
        Project Emoi-Track | Senuvi Jayasinghe | Computer Science
      </footer>
    </div>
  );
}