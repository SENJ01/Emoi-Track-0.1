import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
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

def write_status(status: str, message: str = ""):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump({"status": status, "message": message}, f)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- DYNAMIC DATASET ROUTER ---
def get_emmood_dir(file_path: Path):
    """
    Dynamically detects the correct author dataset based on the story name.
    Raises an error if no matching dataset is found.
    """
    raw_name = file_path.stem.lower()
    story_name = raw_name.replace(".sent", "").strip()

    print(f"🔍 Raw uploaded name: {raw_name}")
    print(f"🔍 Normalized story name: {story_name}")

    for author in os.listdir(DATA_DIR):
        sent_path = DATA_DIR / author / "sent" / f"{story_name}.sent.okpuncs"

        if sent_path.exists():
            emmood_path = DATA_DIR / author / "emmood"
            print(f"✅ Detected author: {author} for story: {story_name}")
            print(f"📂 Using ground truth path: {emmood_path}")
            return emmood_path

    raise FileNotFoundError(
        f"No matching dataset/ground truth directory found for story '{story_name}'."
    )


# ML Pipeline worker
def run_ml_pipeline(input_path: Path):
    start_app_time = time.time()

    try:
        python_executable = sys.executable
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR) + os.pathsep + env.get("PYTHONPATH", "")

        # Step 1: Run narrative inference
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

        # Step 2: Run evaluation using detected ground truth directory
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
                "0.76",
            ],
            check=True,
            cwd=BASE_DIR,
            env=env,
        )

        # Step 3: Inject execution time only after full success
        total_time = round(time.time() - start_app_time, 2)
        latest_path = PREDICTIONS_DIR / "latest_results.json"

        if latest_path.exists():
            with open(latest_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["execution_time"] = total_time

            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

        write_status("done", f"Analysis completed successfully in {total_time}s.")
        print(f"✅ ML pipeline finished successfully in {total_time}s.")

    except Exception as e:
        write_status("error", str(e))
        print("❌ ML pipeline failed:")
        traceback.print_exc()


# API endpoints
@app.get("/")
def root():
    return {"status": "Backend running"}


# Receive uploaded narrative file and save it locally
@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".okpuncs"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a .okpuncs file only.",
            )

        input_path = UPLOAD_DIR / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Clear stale latest outputs before starting a new run
        for path in [
            PREDICTIONS_DIR / "latest_results.json",
            PREDICTIONS_DIR / "latest_trajectory.png",
            PREDICTIONS_DIR / "latest_report.pdf",
        ]:
            if path.exists():
                path.unlink()

        write_status("processing", f"Processing started for {file.filename}")

        thread = threading.Thread(target=run_ml_pipeline, args=(input_path,))
        thread.start()

        return {
            "message": "File uploaded successfully. Processing started.",
            "filename": file.filename,
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        traceback.print_exc()
        write_status("error", f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Upload failed. Check backend logs.")


# Status Endpoint
@app.get("/status/")
def get_status():
    if not STATUS_FILE.exists():
        return {"status": "idle", "message": ""}

    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"status": "error", "message": "Could not read status file."}


@app.get("/results/")
def get_results():
    latest_path = PREDICTIONS_DIR / "latest_results.json"

    if not STATUS_FILE.exists():
        raise HTTPException(status_code=404, detail="No results available.")

    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            status_data = json.load(f)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not read status file.")

    if status_data.get("status") != "done":
        raise HTTPException(status_code=404, detail="No completed results available.")

    if not latest_path.exists():
        raise HTTPException(status_code=404, detail="No results found.")

    reports = list(PREDICTIONS_DIR.glob("full_research_report_*.json"))
    report = {}

    if reports:
        report_path = max(reports, key=os.path.getmtime)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

    with open(latest_path, "r", encoding="utf-8") as f:
        latest = json.load(f)

    csv_files = list(PREDICTIONS_DIR.glob("*_ORIGINAL.csv"))
    latest_csv = max(csv_files, key=os.path.getmtime).name if csv_files else None

    return {
        "summary": latest.get("summary", {}),
        "sentences": latest.get("sentences", []),
        "execution_time": latest.get("execution_time", 0),
        "csv_filename": latest_csv,
        "research": report,
    }


@app.get("/download-csv/{filename}")
def download_csv(filename: str):
    file_path = PREDICTIONS_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")

    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=file_path.name,
    )


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
