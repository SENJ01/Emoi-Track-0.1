import json
from pathlib import Path
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import numpy as np

# DIRECTORY HELPER
def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# SAVE JSON
def save_results_json(output_path: Path, summary: dict, rows: list):
    output_package = {
        "summary": summary,
        "sentences": rows
    }

    with open(output_path, "w", encoding="utf8") as f:
        json.dump(output_package, f, indent=4, default=convert_numpy_types)

# TRAJECTORY PLOT
def generate_emotion_trajectory_plot(rows, output_path: Path):
    sentence_ids = [r["Sentence_ID"] for r in rows]
    nefi_values = [r["NEFI"] for r in rows]
    rupture_flags = [r.get("Rupture_Flag", 0) for r in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(sentence_ids, nefi_values, label="NEFI", color="blue")

    # Mark rupture points
    rupture_points = [
        sentence_ids[i]
        for i in range(len(rupture_flags))
        if rupture_flags[i] == 1
    ]
    rupture_nefi = [
        nefi_values[i]
        for i in range(len(rupture_flags))
        if rupture_flags[i] == 1
    ]

    plt.scatter(rupture_points, rupture_nefi, color="red", label="Rupture")

    plt.xlabel("Sentence Index")
    plt.ylabel("Narrative Energy (NEFI)")
    plt.title("Narrative Emotional Flow Index (NEFI)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# PDF REPORT
def generate_pdf_report(story_name: str, summary: dict, plot_path: Path, output_path: Path):
    doc = SimpleDocTemplate(str(output_path))
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Emoi-Track Analysis Report", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Story: {story_name}", styles["Normal"]))
    elements.append(Paragraph(f"Total Sentences: {summary['total_sentences']}", styles["Normal"]))
    elements.append(Paragraph(f"Average Local Shift: {summary['avg_local_shift']:.4f}", styles["Normal"]))
    elements.append(Paragraph(f"Average NEFI: {summary['avg_nefi']:.4f}", styles["Normal"]))

    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("Emotion Trajectory Graph:", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Image(str(plot_path), width=6 * inch, height=3 * inch))

    doc.build(elements)
