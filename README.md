# EMOi-Track: Narrative Emotion Analysis System

## 📌 Project Overview

EMOi-Track is a narrative emotion analysis system developed to detect:

- **Sentence-level negative emotions** (anger, fear, sadness)
- **Emotional shifts** in narratives
- **Narrative instability** using NEFI (Negative Emotion Fracture Index)

The system integrates:

- **FastAPI** backend
- **React** frontend
- **Fine-tuned RoBERTa-based** emotion model

---

## ⚙️ System Requirements

| Component       | Requirement        |
|-----------------|--------------------|
| Python          | 3.10 or higher     |
| Node.js         | v16+ recommended   |
| Package Manager | npm                |

---

## 📦 Installation Guide

### 1. Install Python Dependencies

Run from the project root:

```bash
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

---

## 🚀 Running the System

### ▶️ Start Backend

```bash
cd backend
uvicorn app:app --reload --port 8000
```

> Backend will run at: **http://127.0.0.1:8000**

### ▶️ Start Frontend

Open a **new terminal**:

```bash
cd frontend
npm run dev
```

> Frontend will run at: **http://localhost:5173**

---

## 🔐 System Access

| Field      | Value     |
|------------|-----------|
| Access Key | `uni2026` |

---

## 📂 Required Folder Structure

Ensure the following directories exist before running the system:

```
data/
uploads/
outputs/predictions/
```

---

## 📊 Input File Requirements

| Property       | Detail           |
|----------------|------------------|
| File format    | `.okpuncs`       |
| Minimum length | 50 sentences     |
| Maximum length | 200 sentences    |

---

## 🧪 How to Use

1. Start the backend server
2. Start the frontend server
3. Open the frontend in your browser (`http://localhost:5173`)
4. Login using the access key
5. Upload a `.okpuncs` narrative file
6. Wait for processing to complete
7. View the results dashboard

---

## ⚠️ Notes

- CPU-based processing — no GPU required
- Paths are portable (no hardcoded paths)
- Outputs are saved in `outputs/predictions/`

---

## 👩‍💻 Author

**Senuvi Jayasinghe**  
Computer Science Undergraduate
