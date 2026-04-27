# Smart Placement and Insights

This project is a complete AI/ML placement website built from the research paper, poster, and bundled dataset.

## Stack

- Frontend: React, HTML, CSS, and JavaScript
- Backend: FastAPI and Uvicorn
- ML: XGBoost model loaded with joblib, plus NumPy/Pandas for inference and dataset analysis
- Data: CSV placement dataset and generated model artifacts

## What the website includes

- Placement prediction form
- Dataset-wide analytics and group trends
- Model metadata and feature transparency
- Role suggestions, readiness scoring, and next steps
- Research summary aligned with the poster and paper
- React-powered single-page frontend served through FastAPI

## Run locally on Windows

The simplest way to start the app from a fresh clone is:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_local.ps1
```

If you prefer to run the steps manually:

1. Open a terminal in the repository root.
2. Create a virtual environment:

```powershell
py -3 -m venv .venv
```

3. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Install the project dependencies:

```powershell
pip install -r requirements.txt
```

5. Start the application:

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

6. Open the app in your browser:

```text
http://127.0.0.1:8000
```

## Notes

- The saved model is bundled in `output/model.pkl` and the dataset is bundled in `output/final_dataset.csv`.
- The repo is fully self-contained for local use; no external model download or preprocessing step is required.
- If the machine has limited disk space, install into a clean environment or a drive with enough free space before running the server.
- The server exposes the prediction API at `/api/predict` and the metadata API at `/api/model-info`.
