import { useState } from "react";
import { api } from "../../api/Client";

export default function UploadZone({ onResults }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return alert("Please select a file.");

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);

    try {
      // Start processing
      await api.post("/analyze/", formData);

      let attempts = 0;

    const interval = setInterval(async () => {
      attempts++;

      const statusRes = await api.get("/status/");

      if (statusRes.data.status === "done") {
        clearInterval(interval);
        const resultsRes = await api.get("/results/");
        onResults(resultsRes.data);
        setLoading(false);
      }

      if (statusRes.data.status === "error") {
        clearInterval(interval);
        alert("Processing failed.");
        setLoading(false);
      }

      // stop polling after 2 minutes just in case
      if (attempts > 60) {
        clearInterval(interval);
        alert("Processing timeout.");
        setLoading(false);
      }

    }, 2000);


    } catch (err) {
      alert("Error analyzing file.");
      console.error(err);
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" accept=".txt" onChange={e => setFile(e.target.files[0])} />
      <button onClick={handleUpload} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>
    </div>
  );
}
