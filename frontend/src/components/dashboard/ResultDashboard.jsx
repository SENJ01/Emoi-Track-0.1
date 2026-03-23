import React from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, Legend, BarChart, Bar, Cell, ReferenceLine, Label,
  PieChart, Pie
} from "recharts";

const COLORS = ["#ef4444", "#3b82f6", "#f59e0b", "#94a3b8"];

export default function ResultDashboard({ data, researcher }) {
if (!data || !data.summary) return <div>Loading Research Data...</div>;
  if (!data || !data.summary) {
    console.log("FULL DATA OBJECT:", data);
    console.log("BEFORE SLIDING:", data.before_sliding);
    console.log("PHASE 2:", data.phase2);
    console.log("SENTENCE 0:", data.sentences?.[0]);
    return <div className="error-state">Data structure mismatch. Check console.</div>;
  }

const { summary, sentences, research, execution_time } = data;
console.log("Research Data Found:", research);
// 2. THE "KITCHEN SINK" DATA TRANSFORMATION
  const chartData = sentences.map((s) => {
    // Check for nested probabilities or flat keys
    const p = s.Probabilities || s.probabilities || s;
    
    // Check for NEFI score variations
    const nefiVal = s.NEFI_Score ?? s.nefi_score ?? s.NEFI ?? s.nefi ?? 0;

    return {
      index: s.Sentence_ID ?? s.sentence_id ?? s.id,
      nefi: nefiVal,
      // Mapping individual emotions from nested or flat structures
      anger: p.anger ?? p.Anger ?? 0,
      fear: p.fear ?? p.Fear ?? 0,
      sadness: p.sadness ?? p.Sadness ?? 0,
      // The red dots rely on this value
      rupture: s.Shift_Label === 1 || s.shift_label === 1 ? nefiVal : null,
    };
  });

  const emotionPieData = Object.entries(summary.emotion_counts || {}).map(([key, value]) => ({
    name: key,
    value: value,
  }));

// Calculate Shift Detection Rate (Dynamic)
  const shiftCount = sentences.filter(s => s.Shift_Label === 1 || s.shift_label === 1).length;
  const shiftRate = sentences.length > 0 ? ((shiftCount / sentences.length) * 100).toFixed(1) : 0;
const shiftRecall = (
    research?.phase2?.nefi_rupture_shift?.report?.["1"]?.recall || 
    research?.phase2?.nefi_rupture_shift?.report?.["1"]?.["recall"] || 
    0
  ).toFixed(3);
// 2. Double-check the Macro-F1 path. 
const macroF1 = (
  research?.before_sliding?.macro_f1 || 
  research?.phase2?.nefi_rupture_shift?.report?.["macro avg"]?.["f1-score"] || 
  0
).toFixed(3);
  const matrix = research?.phase2?.nefi_rupture_shift?.confusion_matrix || [[0, 0], [0, 0]];  // Threshold Calculation (kept for the chart logic, but removed from the top cards)
  const scores = sentences.map(s => s.NEFI_Score ?? s.nefi_score ?? 0);
  const mean = scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
  const variance = scores.length > 0 ? scores.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / scores.length : 0;
  const threshold = mean + Math.sqrt(variance);

  return (
    <div className="dashboard-content">
      
      {/* ================= 1. RESEARCH KPI GRID ================= */}
<div className="stats-grid">
        <Card 
          title="Story Unit" 
          value={summary.story_name || "N/A"} 
          icon="📖" 
        />
        <Card 
          title="Total Samples" 
          value={summary.total_sentences} 
          icon="📊" 
          trend="n-size"
        />
        <Card 
          title="Macro-F1 Score" 
          value={macroF1} 
          icon="🎯" 
          trend="Model Performance" 
        />
        <Card 
          title="Shift Detection Rate" 
          value={`${shiftRate}%`} 
          icon="📈" 
          trend={`${shiftCount} Ruptures`} 
        />
      </div>

      {/* PRIMARY TRAJECTORY */}
      <section className="chart-section">
        <div className="section-header">
          <h2>Narrative Emotional Flow Index (NEFI)</h2>
          <p>Compound instability vs. individual emotion probabilities</p>
        </div>
        <div className="chart-card">
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
              <XAxis dataKey="index" label={{ value: 'Sentence Index', position: 'insideBottom', offset: -5 }} />
              <YAxis yAxisId="left" domain={[0, 1]} label={{ value: 'Probability', angle: -90, position: 'insideLeft' }} />
              <YAxis yAxisId="right" orientation="right" domain={[0, 2]} label={{ value: 'NEFI Score', angle: 90, position: 'insideRight' }} />
              <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
              <Legend verticalAlign="top" height={36}/>
              
              {/* Ensure dataKey matches the transformed chartData keys exactly */}
              <Line yAxisId="left" type="monotone" dataKey="anger" stroke="#ef4444" strokeDasharray="5 5" dot={false} name="Anger Prob" />
              <Line yAxisId="left" type="monotone" dataKey="fear" stroke="#f59e0b" strokeDasharray="5 5" dot={false} name="Fear Prob" />
              <Line yAxisId="right" type="monotone" dataKey="nefi" stroke="#1e293b" strokeWidth={3} dot={false} name="NEFI (Instability)" />
              <Line yAxisId="right" type="monotone" dataKey="rupture" stroke="none" dot={{ r: 5, fill: '#e11d48', strokeWidth: 2 }} name="Rupture Event" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <div className="two-column-layout">
        {/* DISTRIBUTION */}
        <div className="chart-card small">
          <h3>Score Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <XAxis dataKey="index" hide /> 
              <YAxis domain={[0, 2]} />
              <Tooltip />
              <Bar dataKey="nefi" fill="#cbd5e1" />
              <ReferenceLine y={threshold} stroke="#e11d48" strokeDasharray="3 3">
                <Label value="Threshold" position="right" fill="#e11d48" fontSize={10} />
              </ReferenceLine>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* ================= 4. PIE CHART ================= */}
        <div className="chart-card small">
          <h3>Emotion Composition</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={emotionPieData} innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                {emotionPieData.map((entry, index) => <Cell key={index} fill={COLORS[index % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

      {/* <ConfusionMatrix matrix={matrix} /> */}

      </div>

      <section className="table-section">
        <div className="section-header">
          <h2>Sentence-Level Rupture Analysis</h2>
          <p>Highlighting compound emotional shifts</p>
        </div>
        <div className="table-container">
          <table className="academic-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Narrative Excerpt</th>
                <th>Prediction</th>
                <th>NEFI</th>
                <th>Shift</th>
              </tr>
            </thead>
<tbody>
  {sentences.map((s, i) => {
    // 1. Determine the shift status (Matches Rupture_Fl or Shift_Label)
    const isRupture = Number(s.Rupture_Fl ?? s.Shift_Label ?? s.shift_labe) === 1;
    
    // 2. Safely get the prediction
    const prediction = s.Predicted_Emotion ?? s.Predicted_ ?? "UNKNOWN";

    return (
      <tr key={i} className={isRupture ? "rupture-row" : ""}>
        <td>{s.Sentence_ID ?? s.Sentence_ ?? i}</td>
        <td className="text-cell">{s.Text ?? "No Text Available"}</td>
        <td>
          <span className={`badge ${prediction.toLowerCase()}`}>
            {prediction}
          </span>
        </td>
        <td>{Number(s.NEFI ?? s.NEFI_Score ?? 0).toFixed(3)}</td>
        <td>{isRupture ? <span className="r-indicator">1</span> : "0"}</td>
      </tr>
    );
  })}
</tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function ConfusionMatrix({ matrix }) {
  return (
    <div className="chart-card small">
      <h3>Confusion Matrix (Phase 2)</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '16px' }}>
        <div className="matrix-cell">
          <label>True Neg (0,0)</label>
          <div className="matrix-val">{matrix[0][0]}</div>
        </div>
        <div className="matrix-cell highlight-error">
          <label>False Pos (0,1)</label>
          <div className="matrix-val">{matrix[0][1]}</div>
        </div>
        <div className="matrix-cell highlight-error">
          <label>False Neg (1,0)</label>
          <div className="matrix-val">{matrix[1][0]}</div>
        </div>
        <div className="matrix-cell highlight-success">
          <label>True Pos (1,1)</label>
          <div className="matrix-val">{matrix[1][1]}</div>
        </div>
      </div>
    </div>
  );
}

function Card({ title, value, icon, trend }) {
  // Check if value is a string and handle potential long titles
  const isLongTitle = typeof value === 'string' && value.length > 12;

  return (
    <div className="stat-card" style={{ 
      padding: '16px', 
      background: '#fff', 
      borderRadius: '12px', 
      border: '1px solid #e2e8f0',
      minHeight: '110px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      overflow: 'hidden' // Essential to stop the breakout
    }}>
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
        <span className="card-icon" style={{ fontSize: '18px' }}>{icon}</span>
        <span className="card-trend" style={{ fontSize: '10px', color: '#94a3b8', fontWeight: '600' }}>{trend}</span>
      </div>
      <div className="card-body">
        <h4 style={{ margin: 0, fontSize: '11px', color: '#64748b', textTransform: 'uppercase' }}>{title}</h4>
        <p style={{ 
          margin: '4px 0 0', 
          fontSize: isLongTitle ? '20px' : '20px', // Shrinks font for 'ginger_and_pickles'
          fontWeight: 'bold',
          color: '#1e293b',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          cursor: 'help'
        }} title={value}>
          {value}
        </p>
      </div>
    </div>
  );
}
