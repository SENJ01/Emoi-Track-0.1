import os
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from fpdf import FPDF
import shutil
import subprocess
import threading
import sys
import traceback
import json

app = FastAPI()

# Base Project Directory
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATUS_FILE = BASE_DIR / "status.txt"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"
DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- DYNAMIC DATASET ROUTER ---
def get_emmood_dir(file_path: Path):
    """
    Automatically routes the ground-truth path based on the story name
    or its original location.
    """
    path_str = str(file_path).lower()

    # Check for Grimms
    if "grimms" in path_str or "snowdrop" in path_str:
        return DATA_DIR / "Grimms" / "emmood"

    # Check for HCAndersen
    elif "hcandersen" in path_str or "andersen" in path_str:
        return DATA_DIR / "HCAndersen" / "emmood"

    # Default to Potter
    else:
        # Note: Move your Potter emmood files to: /data/Potter/emmood/
        return DATA_DIR / "Potter" / "emmood"


# --- ML PIPELINE WORKER ---
def run_ml_pipeline(input_path: Path):
    start_app_time = time.time()
    try:
        python_executable = sys.executable
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR) + os.pathsep + env.get("PYTHONPATH", "")

        # Step 1: Run Inference (EMOi-Track Core)
        subprocess.run(
            [
                python_executable,
                "-m",
                "ml.inference.predict_narrative",
                "--taxonomy",
                "negative_emo_roberta",
                "--input_file",
                str(input_path),
            ],
            check=True,
            cwd=BASE_DIR,
            env=env,
        )

        # Step 2: Evaluation (Synchronized Threshold & Dynamic Path)
        selected_emmood_dir = get_emmood_dir(input_path)
        print(f"DEBUG: Selected Ground Truth Path: {selected_emmood_dir}")

        subprocess.run(
            [
                python_executable,
                "-m",
                "ml.evaluation.final_validation",
                "--input_file",
                str(input_path),
                "--emmood_dir",
                str(selected_emmood_dir),
                "--threshold",
                "0.88",  # Fixed to match your research requirements
            ],
            check=True,
            cwd=BASE_DIR,
            env=env,
        )

        # Step 3: Performance Metadata Injection
        total_time = round(time.time() - start_app_time, 2)
        latest_path = PREDICTIONS_DIR / "latest_results.json"

        if latest_path.exists():
            with open(latest_path, "r") as f:
                data = json.load(f)
            data["execution_time"] = total_time
            with open(latest_path, "w") as f:
                json.dump(data, f, indent=4)

        STATUS_FILE.write_text("done")
        print(f"✅ ML pipeline finished successfully in {total_time}s.")

    except Exception:
        STATUS_FILE.write_text("error")
        print("❌ ML pipeline failed:")
        traceback.print_exc()


# API endpoints
@app.get("/")
def root():
    return {"status": "Backend running"}


# Upload Endpoint
@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    try:
        input_path = UPLOAD_DIR / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        STATUS_FILE.write_text("processing")

        thread = threading.Thread(target=run_ml_pipeline, args=(input_path,))
        thread.start()

        return {
            "message": "File uploaded successfully. Processing started.",
            "filename": file.filename,
        }

    except Exception:
        traceback.print_exc()
        return {"error": "Upload failed. Check backend logs."}


# Status Endpoint
@app.get("/status/")
def get_status():
    if not STATUS_FILE.exists():
        return {"status": "idle"}
    return {"status": STATUS_FILE.read_text()}


@app.get("/results/")
def get_results():
    # Path to the latest summary
    latest_path = PREDICTIONS_DIR / "latest_results.json"

    if not latest_path.exists():
        return {"error": "No results found. Run analysis first."}

    # Get all research reports in the folder
    reports = list(PREDICTIONS_DIR.glob("full_research_report_*.json"))

    if not reports:
        # Fallback: Just return what's in latest_results if no report exists
        with open(latest_path) as f:
            latest = json.load(f)
        return {
            "summary": latest["summary"],
            "sentences": latest["sentences"],
            "research": {},
        }

    report_path = max(reports, key=os.path.getmtime)

    with open(latest_path) as f1, open(report_path) as f2:
        latest = json.load(f1)
        report = json.load(f2)

    return {
        "summary": latest["summary"],
        "sentences": latest["sentences"],
        "execution_time": latest.get("execution_time", 0),
        "research": report,
    }


@app.get("/plot/")
def get_plot():
    # Find all png files in the predictions dir
    plots = list(PREDICTIONS_DIR.glob("*.png"))
    if not plots:
        return {"error": "No plot found"}

    # Return the newest one
    latest_plot = max(plots, key=os.path.getmtime)
    return FileResponse(latest_plot, media_type="image/png")


@app.get("/report/")
def generate_and_download_report():
    # Get the latest results JSON
    latest_path = PREDICTIONS_DIR / "latest_results.json"
    if not latest_path.exists():
        return {"error": "No results found to generate a report."}

    with open(latest_path) as f:
        data = json.load(f)

    # Get the time from the loaded 'data'
    exec_time = data.get("execution_time", "N/A")

    summary = data.get("summary", {})

    # Create the PDF in memory
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 10, "EMOI-TRACK: Emotion Analysis Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Total Processing Time: {exec_time} seconds", ln=True)
    pdf.ln(5)

    # Summary Section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Story: {summary.get('story_name', 'Unknown')}", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Total Sentences: {summary.get('total_sentences')}", ln=True)
    pdf.cell(
        0, 10, f"Average Local Shift: {summary.get('avg_local_shift', 0):.4f}", ln=True
    )
    pdf.cell(
        0,
        10,
        f"Average Anchor Shift: {summary.get('avg_anchor_shift', 0):.4f}",
        ln=True,
    )
    pdf.ln(10)

    # 3. Save the PDF to a temporary file
    report_path = PREDICTIONS_DIR / "generated_report.pdf"
    pdf.output(str(report_path))

    # 4. Serve the file to the user
    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=f"Report_{summary.get('story_name', 'analysis')}.pdf",
    )
