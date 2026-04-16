import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
  Cell,
  ReferenceLine,
  Label,
  LabelList,
  PieChart,
  Pie,
} from "recharts";

const COLORS = ["#131969", "#3b82f6", "#1d4ed8", "#8eaccb"];

export default function ResultDashboard({ data, researcher, executionTime }) {
  if (!data || !data.summary || !data.sentences) {
    return <div>Loading Research Data...</div>;
  }

  const { summary, sentences, research } = data;

  const chartData = sentences.map((s, i) => {
    const p = s.Probabilities || s.probabilities || s;
    const nefiVal = Number(
      s.NEFI_Score ?? s.nefi_score ?? s.NEFI ?? s.nefi ?? 0,
    );
    const ruptureFlag = Number(
      s.Rupture_Flag ?? s.Rupture_Fl ?? s.rupture_flag ?? 0,
    );

    const prediction = (
      s.Predicted_Emotion ??
      s.Predicted_ ??
      s.predicted_emotion ??
      "UNKNOWN"
    ).toUpperCase();

    return {
      index: s.Sentence_ID ?? s.sentence_id ?? i,
      nefi: nefiVal,
      anger: Number(p.anger ?? p.Anger ?? 0),
      fear: Number(p.fear ?? p.Fear ?? 0),
      sadness: Number(p.sadness ?? p.Sadness ?? 0),
      rupture: ruptureFlag === 1 ? nefiVal : null,
      prediction,
    };
  });

  const emotionPieData = Object.entries(summary.emotion_counts || {}).map(
    ([key, value]) => ({
      name: key,
      value,
    }),
  );

  const ruptureCount = sentences.filter(
    (s) => Number(s.Rupture_Flag ?? s.Rupture_Fl ?? s.rupture_flag ?? 0) === 1,
  ).length;

  const macroF1 = (research?.phase1?.before_sliding?.macro_f1 || 0).toFixed(3);

  const accuracy = (research?.phase1?.before_sliding?.accuracy || 0).toFixed(3);

  const unknownF1 = (
    research?.phase1?.before_sliding?.report?.unknown?.["f1-score"] || 0
  ).toFixed(3);

  const precisionAt5 = (
    research?.phase2?.top5_nefi_shift?.precision_at_5 || 0
  ).toFixed(3);

  const top5Matches = research?.phase2?.top5_nefi_shift?.matches_top5 ?? 0;

  const methodComparisonData = [
    {
      method: "Label",
      f1: Number(research?.phase2?.label_shift?.f1 || 0),
    },
    {
      method: "Local",
      f1: Number(research?.phase2?.local_distance_shift?.f1 || 0),
    },
    {
      method: "Angle",
      f1: Number(research?.phase2?.trajectory_angle_shift?.f1 || 0),
    },
    {
      method: "Prob",
      f1: Number(research?.phase3?.probability_shift?.f1 || 0),
    },
  ];

  const scores = chartData.map((s) => s.nefi);
  const mean =
    scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;

  const variance =
    scores.length > 0
      ? scores.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) /
        scores.length
      : 0;

  const threshold = mean + Math.sqrt(variance);

  const labelCM = research?.phase2?.label_shift?.confusion_matrix || [
    [0, 0],
    [0, 0],
  ];
  const probCM = research?.phase3?.probability_shift?.confusion_matrix || [
    [0, 0],
    [0, 0],
  ];
  const nefiCM = research?.phase2?.nefi_rupture_shift?.confusion_matrix || [
    [0, 0],
    [0, 0],
  ];

  return (
    <div className="dashboard-content">
      <div className="stats-grid">
        <Card
          title="Narrative Summary"
          metrics={[
            { label: "STORY", value: summary.story_name || "N/A" },
            { label: "SENTENCES", value: summary.total_sentences },
            { label: "LATENCY", value: `${executionTime || "0.00"}s` },
          ]}
          icon="📖"
          color="#18223e"
          bg="#ffffff"
        />

        <Card
          title="Classification"
          metrics={[
            { label: "Macro F1", value: macroF1 },
            { label: "Accuracy", value: accuracy },
            { label: "UNKNOWN F1", value: unknownF1 },
          ]}
          icon="🧠"
          color="#18223e"
          bg="#ffffff"
        />

        <Card
          title="Shift Detection"
          metrics={[
            {
              label: "Label F1",
              value: (research?.phase2?.label_shift?.f1 || 0).toFixed(3),
            },
            {
              label: "Trajectory F1",
              value: (
                research?.phase2?.trajectory_angle_shift?.f1 || 0
              ).toFixed(3),
            },
            {
              label: "Probability F1",
              value: (research?.phase3?.probability_shift?.f1 || 0).toFixed(3),
            },
          ]}
          subtitle="Per-story F1 comparison across methods"
          icon="📈"
          color="#18223e"
          bg="#ffffff"
        />

        <Card
          title="NEFI Rupture Detection"
          metrics={[
            { label: "PRECISION@5", value: precisionAt5 },
            { label: "TOP-K MATCHES", value: `${top5Matches}/5` },
            { label: "RUPTURE COUNT", value: ruptureCount },
          ]}
          icon="⚡"
          color="#10b981"
          bg="#ffffff"
        />
      </div>

      <section className="chart-section">
        <div className="section-header">
          <h2 style={{ color: "#2c3281" }}>
            Narrative Emotional Fracture Index (NEFI)
          </h2>
          <p>
            Emotion probabilities and rupture-focused instability across the
            narrative
          </p>
        </div>

        <div className="chart-card">
          <ResponsiveContainer width="100%" height={420}>
            <LineChart data={chartData}>
              <CartesianGrid
                strokeDasharray="3 3"
                vertical={false}
                stroke="#f1f5f9"
              />
              <XAxis
                dataKey="index"
                label={{
                  value: "Sentence Index",
                  position: "insideBottom",
                  offset: -5,
                }}
              />
              <YAxis
                yAxisId="left"
                domain={[0, 1]}
                label={{
                  value: "Probability",
                  angle: -90,
                  position: "insideLeft",
                }}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                domain={[0, 2]}
                label={{
                  value: "NEFI Score",
                  angle: 90,
                  position: "insideRight",
                }}
              />
              <Tooltip content={<CustomNEFITooltip />} />
              <Legend
                verticalAlign="top"
                align="center"
                iconType="circle"
                wrapperStyle={{
                  paddingBottom: "6px",
                  fontSize: "13px",
                  lineHeight: "1.2",
                }}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="anger"
                stroke="#224776"
                dot={false}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="fear"
                stroke="#3b82f6"
                dot={false}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="sadness"
                stroke="#1d4ed8"
                dot={false}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="nefi"
                stroke="#0c1d4c"
                strokeWidth={2.5}
                dot={false}
                name="NEFI"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="rupture"
                stroke="none"
                dot={{ r: 5, fill: "#ff0022", strokeWidth: 2 }}
                name="Rupture Event"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <div className="two-column-layout">
        <div className="chart-card small">
          <h3 style={{ color: "#253f78" }}>Emotion Composition</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={emotionPieData}
                innerRadius={55}
                outerRadius={82}
                paddingAngle={4}
                dataKey="value"
                nameKey="name"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}`}
              >
                {emotionPieData.map((entry, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card small">
          <h3 style={{ color: "#201a7b" }}>NEFI Evaluation</h3>
          <div
            style={{ lineHeight: "1.9", color: "#475569", fontSize: "14px" }}
          >
            <p>
              <strong>Precision@5:</strong> {precisionAt5}
            </p>
            <p>
              <strong>Top-5 Matches:</strong> {top5Matches} / 5
            </p>
            <p>
              <strong>Rupture Count:</strong> {ruptureCount}
            </p>
            <p>
              <strong>Interpretation:</strong> NEFI prioritises high-intensity
              rupture points rather than all transitions.
            </p>
          </div>
        </div>
      </div>
      <p style={{ fontSize: "14px", color: "#64748b", marginBottom: "15px" }}>
        **Confusion matrices are shown for representative methods: baseline
        (Label), strongest alternative (Probability), and the proposed
        NEFI-based rupture detection.
      </p>
      <div
        className="three-column-layout"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "20px",
          marginBottom: "24px",
        }}
      >
        <ConfusionMatrix
          title="Label Shift (Baseline)"
          cm={labelCM}
          color="#5b7da4"
        />

        <ConfusionMatrix
          title="Probability Shift"
          cm={probCM}
          color="#253f78"
        />

        <ConfusionMatrix title="NEFI" cm={nefiCM} color="#0c1d4c" />
      </div>

      <div className="two-column-layout">
        <div className="chart-card small">
          <h3 style={{ color: "#0e1163" }}>NEFI Threshold Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <XAxis dataKey="index" hide />
              <YAxis domain={[0, 2]} />
              <Tooltip />
              <Bar dataKey="nefi" fill="#bfdbfe" />
              <ReferenceLine
                y={threshold}
                stroke="#1d4ed8"
                strokeDasharray="3 3"
              >
                <Label
                  value="μ + σ Threshold"
                  position="right"
                  fill="#1d4ed8"
                  fontSize={10}
                />
              </ReferenceLine>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card small">
          <h3 style={{ color: "#2563eb" }}>Method Interpretation</h3>
          <div
            style={{ lineHeight: "1.9", color: "#475569", fontSize: "14px" }}
          >
            <p>
              <strong>Selective Prediction:</strong> low-confidence segments are
              assigned UNKNOWN.
            </p>
            <p>
              <strong>Best shift signal:</strong> probability-based detection
              achieved the strongest shift F1 in this evaluation.
            </p>
            <p>
              <strong>NEFI = √(C² + D² + S²)</strong>
            </p>
            <p>
              <strong>C:</strong> trajectory angle
            </p>
            <p>
              <strong>D:</strong> momentum deviation
            </p>
            <p>
              <strong>S:</strong> label shift
            </p>
            <p>
              <strong>Threshold:</strong> μ + σ of NEFI values
            </p>
          </div>
        </div>
        {/* <div className="chart-card small">
          <h3 style={{ color: "#253f78" }}>Emotion Composition</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={emotionPieData}
                innerRadius={60}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
              >
                {emotionPieData.map((entry, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div> */}
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
                const isRupture =
                  Number(s.Rupture_Fl ?? s.Shift_Label ?? s.shift_labe) === 1;

                // 2. Safely get the prediction
                const prediction =
                  s.Predicted_Emotion ?? s.Predicted_ ?? "UNKNOWN";

                return (
                  <tr key={i} className={isRupture ? "rupture-row" : ""}>
                    <td>{s.Sentence_ID ?? s.Sentence_ ?? i}</td>
                    <td className="text-cell">
                      {s.Text ?? "No Text Available"}
                    </td>
                    <td>
                      <span className={`badge ${prediction.toLowerCase()}`}>
                        {prediction}
                      </span>
                    </td>
                    <td>{Number(s.NEFI ?? s.NEFI_Score ?? 0).toFixed(3)}</td>
                    <td>
                      {isRupture ? <span className="r-indicator">1</span> : "0"}
                    </td>
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

function Card({
  title,
  value,
  subtitle,
  metrics = [],
  icon,
  color = "#1e293b",
  bg = "#ffffff",
}) {
  const heading = title || icon;
  const iconOnly = title ? icon : null;
  const hasMainValue = value !== undefined && value !== null && value !== "";
  const isLongValue = typeof value === "string" && value.length > 18;

  return (
    <div
      className="stat-card"
      style={{
        padding: "18px",
        background: bg,
        borderRadius: "14px",
        border: `1px solid ${color}22`,
        borderTop: `5px solid ${color}`,
        minHeight: "220px",
        display: "flex",
        flexDirection: "column",
        justifyContent: "flex-start",
        overflow: "hidden",
        boxShadow: "0 4px 14px rgba(15, 23, 42, 0.04)",
      }}
    >
      <div
        className="card-header"
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          marginBottom: "14px",
          color,
          fontSize: "13px",
          fontWeight: "800",
          textTransform: "uppercase",
          letterSpacing: "0.4px",
          lineHeight: 1.3,
        }}
      >
        {iconOnly && <span style={{ fontSize: "20px" }}>{iconOnly}</span>}
        <span>{heading}</span>
      </div>

      <div
        className="card-body"
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "12px",
          flex: 1,
        }}
      >
        {hasMainValue && (
          <div
            style={{
              marginBottom: metrics.length > 0 ? "2px" : "6px",
            }}
          >
            <p
              style={{
                margin: "0 0 10px",
                fontSize: isLongValue ? "17px" : "22px",
                fontWeight: "800",
                color: "#0f172a",
                whiteSpace: "normal",
                wordBreak: "break-word",
                lineHeight: "1.4",
              }}
              title={String(value)}
            >
              {value}
            </p>
          </div>
        )}

        {metrics.length > 0 ? (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "10px",
              marginTop: hasMainValue ? "4px" : "0",
            }}
          >
            {metrics.map((m, idx) => (
              <div
                key={idx}
                style={{
                  padding: "12px 14px",
                  borderRadius: "10px",
                  background: "#f8fafc",
                  border: `1px solid ${color}22`,
                  borderLeft: `4px solid ${color}`,
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  gap: "12px",
                }}
              >
                <span
                  style={{
                    fontSize: "12px",
                    fontWeight: "700",
                    color: "#64748b",
                    textTransform: "uppercase",
                    letterSpacing: "0.3px",
                    lineHeight: 1.3,
                    minWidth: "78px",
                  }}
                >
                  {m.label}
                </span>

                <span
                  style={{
                    fontSize: "16px",
                    fontWeight: "800",
                    color: "#0f172a",
                    lineHeight: "1.35",
                    whiteSpace: "normal",
                    wordBreak: "break-word",
                    overflowWrap: "anywhere",
                    textAlign: "right",
                    flex: 1,
                  }}
                >
                  {m.value}
                </span>
              </div>
            ))}
          </div>
        ) : subtitle ? (
          <p
            style={{
              margin: 0,
              fontSize: "13px",
              color: "#64748b",
              fontWeight: "600",
              lineHeight: "1.5",
            }}
          >
            {subtitle}
          </p>
        ) : null}
      </div>
    </div>
  );
}

function getConfusionMatrixNote(tn, fp, fn, tp) {
  const total = tn + fp + fn + tp;
  if (total === 0) {
    return "No evaluation data available for this run.";
  }

  if (fp === 0 && fn === 0 && tp > 0) {
    return "The method cleanly separates rupture and non-rupture cases in this run, with no observed false alarms or missed positives.";
  }

  if (tp === 0 && fp === 0) {
    return "The method adopts a highly conservative strategy, avoiding false alarms but not identifying any positive rupture events in this run.";
  }

  if (tp === 0 && fp > 0) {
    return "The method struggles to recover true rupture events in this setting, while still producing some false positives, indicating instability in positive detection.";
  }

  if (fp === 0 && tp > 0) {
    return "The method demonstrates strong precision, identifying rupture events without false alarms, though some positives may still be missed.";
  }

  if (fn === 0 && tp > 0) {
    return "The method achieves high sensitivity, successfully recovering all rupture events, though this may come with some over-prediction.";
  }

  if (fp >= 2 * tp && fp > fn) {
    return "The method favours sensitivity, capturing rupture events but with a noticeable increase in false-positive predictions.";
  }

  if (fn >= 2 * tp && fn > fp) {
    return "The method leans toward conservative detection, reducing false alarms but missing a number of true rupture events.";
  }

  if (tn > tp && fn > fp) {
    return "The method performs more reliably on non-rupture cases, with missed positives being the primary limitation.";
  }

  if (tp > tn && fp > fn) {
    return "The method emphasises rupture detection, capturing more positive events, though at the cost of increased false positives.";
  }

  if (fp > fn) {
    return "The method shows higher sensitivity, capturing more rupture events but introducing additional false positives.";
  }

  if (fn > fp) {
    return "The method is relatively conservative, prioritising fewer false alarms while missing some rupture events.";
  }

  return "The method shows a balanced detection pattern across rupture and non-rupture cases.";
}

function ConfusionMatrix({ title, cm, color = "#1d4ed8", note }) {
  const tn = cm?.[0]?.[0] ?? 0;
  const fp = cm?.[0]?.[1] ?? 0;
  const fn = cm?.[1]?.[0] ?? 0;
  const tp = cm?.[1]?.[1] ?? 0;

  const autoNote = getConfusionMatrixNote(tn, fp, fn, tp);

  return (
    <div
      className="chart-card small"
      style={{
        borderTop: `4px solid ${color}`,
      }}
    >
      <h3 style={{ color, marginBottom: "14px" }}>{title} Confusion Matrix</h3>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, 1fr)",
          gap: "8px",
          marginBottom: "12px",
        }}
      >
        <div
          style={{
            padding: "12px",
            borderRadius: "10px",
            background: "#dbeafe",
            color: "#1e3a8a",
            textAlign: "center",
            fontWeight: "700",
            fontSize: "13px",
          }}
        >
          TN
          <br />
          {tn}
        </div>
        <div
          style={{
            padding: "12px",
            borderRadius: "10px",
            background: "#eff6ff",
            color: "#2563eb",
            textAlign: "center",
            fontWeight: "700",
            fontSize: "13px",
          }}
        >
          FP
          <br />
          {fp}
        </div>
        <div
          style={{
            padding: "12px",
            borderRadius: "10px",
            background: "#e0f2fe",
            color: "#1d4ed8",
            textAlign: "center",
            fontWeight: "700",
            fontSize: "13px",
          }}
        >
          FN
          <br />
          {fn}
        </div>
        <div
          style={{
            padding: "12px",
            borderRadius: "10px",
            background: "#bfdbfe",
            color: "#0c1d4c",
            textAlign: "center",
            fontWeight: "700",
            fontSize: "13px",
          }}
        >
          TP
          <br />
          {tp}
        </div>
      </div>

      <p
        style={{
          margin: 0,
          fontSize: "12px",
          color: "#475569",
          lineHeight: "1.6",
          fontWeight: "600",
        }}
      >
        {note ?? autoNote}
      </p>
    </div>
  );
}

function CustomNEFITooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;

  const point = payload[0]?.payload || {};

  const anger = Number(point.anger || 0);
  const fear = Number(point.fear || 0);
  const sadness = Number(point.sadness || 0);
  const nefi = Number(point.nefi || 0);
  const isRupture = point.rupture !== null && point.rupture !== undefined;
  const prediction = point.prediction || "UNKNOWN";

  let dominantEmotion = "UNKNOWN";
  const maxEmotion = Math.max(anger, fear, sadness);

  if (maxEmotion > 0) {
    if (anger === maxEmotion) dominantEmotion = "ANGER";
    else if (fear === maxEmotion) dominantEmotion = "FEAR";
    else dominantEmotion = "SADNESS";
  } else {
    dominantEmotion = prediction;
  }

  return (
    <div
      style={{
        background: "#ffffff",
        border: "1px solid #d1d5db",
        padding: "10px 12px",
        borderRadius: "6px",
        minWidth: "185px",
      }}
    >
      <p style={{ margin: "0 0 6px", fontWeight: "700", color: "#0f172a" }}>
        {label}
      </p>

      <p
        style={{
          margin: "4px 0",
          color: "#0c1d4c",
          fontSize: "13px",
          fontWeight: "700",
        }}
      >
        NEFI: {nefi.toFixed(4)}
      </p>

      <p
        style={{
          margin: "4px 0",
          color: "#172c43",
          fontSize: "13px",
          fontWeight: "700",
        }}
      >
        Emotion: {dominantEmotion}
      </p>

      {isRupture && (
        <p
          style={{
            margin: "4px 0",
            color: "#1d4ed8",
            fontSize: "13px",
            fontWeight: "700",
          }}
        >
          Rupture Event: Yes
        </p>
      )}
    </div>
  );
}
