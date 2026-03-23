import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import UploadPage from "./pages/UploadPage";
import ResultPage from "./pages/ResultPage";
import Login from "./components/login/Login";
import SplashScreen from "./components/splashscreen/SplashScreen";

export default function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [researcherName, setResearcherName] = useState('');

  // State to store the time retrieved from the backend
  const [executionTime, setExecutionTime] = useState(null);

  // Function to handle login
  const handleAuth = (status, name) => {
    setIsAuthenticated(status);
    setResearcherName(name);
  };

  // Show Splash Screen first
  if (showSplash) {
    return <SplashScreen onFinished={() => setShowSplash(false)} />;
  }

  // Once Splash is done, show the App Routes
  return (
    <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes>
        {/* Login Route - Pass handleAuth to capture name */}
        <Route 
          path="/login" 
          element={!isAuthenticated ? <Login setAuth={handleAuth} /> : <Navigate to="/" />} 
        />

        {/* Protected Root/Upload Route - Passes researcher name */}
        <Route 
          path="/" 
          element={isAuthenticated ? <UploadPage researcher={researcherName} setExecutionTime={setExecutionTime}/> : <Navigate to="/login" />} 
        />

        {/* Protected Results Route - Passes researcher name */}
        <Route 
          path="/results" 
          element={isAuthenticated ? <ResultPage researcher={researcherName} executionTime={executionTime}/> : <Navigate to="/login" />} 
        />

        {/* Catch-all: Redirect unknown paths */}
        <Route path="*" element={<Navigate to={isAuthenticated ? "/" : "/login"} />} />
      </Routes>
    </Router>
  );
}