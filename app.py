from collections import Counter, defaultdict
from csv import DictReader
from pathlib import Path
from statistics import mean

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "output" / "model.pkl"
DATASET_PATH = BASE_DIR / "output" / "final_dataset.csv"
FEATURE_NAMES = [
    "CGPA",
    "Internships",
    "Projects",
    "Certifications",
    "Communication_Skills",
    "Aptitude_Score",
    "Backlogs",
]

PROJECT_OVERVIEW = {
    "title": "Smart Placement and Insights",
    "summary": (
        "A placement analytics website that combines a trained machine learning model "
        "with dataset-level insights from the research paper and poster."
    ),
    "objectives": [
        "Predict whether a student is likely to be placed",
        "Surface the strongest placement drivers from the dataset",
        "Provide skill-gap guidance and role suggestions",
        "Package the research into a clean, interactive website",
    ],
    "tech_stack": {
        "frontend": ["React", "HTML", "CSS", "JavaScript"],
        "backend": ["FastAPI", "Uvicorn"],
        "ml": ["scikit-learn", "XGBoost", "joblib", "NumPy"],
        "data": ["CSV dataset", "placement research notebooks"],
    },
    "features": [
        "Placement prediction form",
        "Dataset-wide placement analytics",
        "Model confidence and readiness score",
        "Role recommendations and next steps",
        "React-powered responsive single-page interface",
    ],
}

app = FastAPI(title="Smart Placement and Insights", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


class FallbackModel:
    classes_ = np.array([0, 1])

    def predict(self, features):
        row = np.asarray(features, dtype=float)[0]
        cgpa, internships, projects, certifications, communication, aptitude, backlogs = row
        score = (
            cgpa * 11.0
            + internships * 8.0
            + projects * 6.0
            + certifications * 4.0
            + communication * 0.18
            + aptitude * 0.22
            - backlogs * 10.0
        )
        return np.array([1 if score >= 100 else 0])

    def predict_proba(self, features):
        row = np.asarray(features, dtype=float)[0]
        cgpa, internships, projects, certifications, communication, aptitude, backlogs = row
        score = (
            cgpa * 11.0
            + internships * 8.0
            + projects * 6.0
            + certifications * 4.0
            + communication * 0.18
            + aptitude * 0.22
            - backlogs * 10.0
        )
        probability = 1.0 / (1.0 + np.exp(-(score - 100.0) / 10.0))
        return np.array([[1.0 - probability, probability]])


def load_rows():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {DATASET_PATH}")
    with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(DictReader(handle))


def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def is_placed(value):
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "placed", "pass", "y"}


def average(values):
    values = [value for value in values if value is not None]
    return round(mean(values), 2) if values else 0.0


def build_insights(rows):
    total_students = len(rows)
    placement_counts = Counter()
    branch_stats = defaultdict(lambda: {"total": 0, "placed": 0})
    tier_stats = defaultdict(lambda: {"total": 0, "placed": 0})
    numeric_groups = defaultdict(lambda: {"placed": [], "not_placed": []})

    numeric_columns = [
        "CGPA",
        "Internships",
        "Projects",
        "Certifications",
        "Programming_Skills",
        "Aptitude_Score",
        "Communication_Skills",
        "logical_reasoning_score",
        "Hackathons",
        "github_repos",
        "linkedin_connections",
        "mock_interview_score",
        "Attendance",
        "Backlogs",
        "extracurricular_score",
        "Leadership",
        "volunteer_experience",
        "sleep_hours",
        "study_hours_per_day",
        "10th_Percentage",
        "12th_Percentage",
        "DSA_Score",
        "Teamwork",
        "Skill_Score",
        "Experience_Score",
        "Academic_Consistency",
    ]

    for row in rows:
        placed = is_placed(row.get("Placement", row.get("PlacementStatus", "")))
        status_key = "Placed" if placed else "Not Placed"
        placement_counts[status_key] += 1

        branch = row.get("branch") or row.get("Core_Subjects") or "Unknown"
        tier = row.get("college_tier") or row.get("company_type") or "Unknown"
        branch_stats[branch]["total"] += 1
        tier_stats[tier]["total"] += 1
        if placed:
            branch_stats[branch]["placed"] += 1
            tier_stats[tier]["placed"] += 1

        for column in numeric_columns:
            value = to_float(row.get(column))
            if value is None:
                continue
            numeric_groups[column]["placed" if placed else "not_placed"].append(value)

    total_placed = placement_counts["Placed"]
    total_not_placed = placement_counts["Not Placed"]
    placement_rate = round((total_placed / total_students) * 100, 2) if total_students else 0.0

    branch_rows = []
    for branch, stats in branch_stats.items():
        if stats["total"] < 500:
            continue
        branch_rows.append(
            {
                "label": branch,
                "placement_rate": round((stats["placed"] / stats["total"]) * 100, 2),
                "students": stats["total"],
            }
        )
    branch_rows.sort(key=lambda item: item["placement_rate"], reverse=True)

    tier_rows = [
        {
            "label": tier,
            "placement_rate": round((stats["placed"] / stats["total"]) * 100, 2),
            "students": stats["total"],
        }
        for tier, stats in tier_stats.items()
        if stats["total"]
    ]
    tier_rows.sort(key=lambda item: item["placement_rate"], reverse=True)

    driver_rows = []
    for column, groups in numeric_groups.items():
        placed_values = groups["placed"]
        not_placed_values = groups["not_placed"]
        if len(placed_values) < 10 or len(not_placed_values) < 10:
            continue
        placed_avg = average(placed_values)
        not_placed_avg = average(not_placed_values)
        delta = round(placed_avg - not_placed_avg, 2)
        driver_rows.append(
            {
                "label": column,
                "placed_avg": placed_avg,
                "not_placed_avg": not_placed_avg,
                "delta": delta,
            }
        )

    driver_rows.sort(key=lambda item: abs(item["delta"]), reverse=True)

    return {
        "summary": {
            "total_students": total_students,
            "placed_students": total_placed,
            "not_placed_students": total_not_placed,
            "placement_rate": placement_rate,
            "avg_cgpa": average(numeric_groups["CGPA"]["placed"] + numeric_groups["CGPA"]["not_placed"]),
            "avg_aptitude": average(numeric_groups["Aptitude_Score"]["placed"] + numeric_groups["Aptitude_Score"]["not_placed"]),
            "avg_communication": average(numeric_groups["Communication_Skills"]["placed"] + numeric_groups["Communication_Skills"]["not_placed"]),
            "avg_projects": average(numeric_groups["Projects"]["placed"] + numeric_groups["Projects"]["not_placed"]),
            "avg_internships": average(numeric_groups["Internships"]["placed"] + numeric_groups["Internships"]["not_placed"]),
        },
        "top_branches": branch_rows[:6],
        "college_tiers": tier_rows,
        "key_drivers": driver_rows[:6],
    }


def readiness_score(data):
    score = (
        min(data.CGPA, 10.0) * 8.0
        + min(data.Internships, 6.0) * 7.0
        + min(data.Projects, 8.0) * 6.0
        + min(data.Certifications, 8.0) * 4.5
        + min(data.Communication_Skills, 100.0) * 0.2
        + min(data.Aptitude_Score, 100.0) * 0.22
        - min(data.Backlogs, 10.0) * 8.0
    )
    return round(max(0.0, min(100.0, score)), 1)


def role_recommendation(data):
    signals = []
    if data.CGPA >= 8.2 and data.Projects >= 3:
        signals.append("Software development")
    if data.Aptitude_Score >= 80 and data.Communication_Skills >= 70:
        signals.append("Business analyst")
    if data.Internships >= 2 and data.Certifications >= 2:
        signals.append("Industry-ready internship track")
    if data.Backlogs > 0:
        signals.append("Priority: clear backlogs")
    if not signals:
        signals.append("Foundational placement support")
    return signals


def build_model_info(model):
    algorithm = type(model).__name__
    info = {
        "algorithm": algorithm,
        "supports_probability": hasattr(model, "predict_proba"),
        "feature_names": FEATURE_NAMES,
        "target": "Placement",
        "input_size": len(FEATURE_NAMES),
        "notes": [
            "The model consumes the same seven core features shown on the website.",
            "Dataset analytics are computed from the bundled placement data file.",
        ],
    }

    if hasattr(model, "feature_importances_"):
        importances = list(getattr(model, "feature_importances_"))
        ranked = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda item: item[1],
            reverse=True,
        )
        info["top_importances"] = [
            {"label": name, "score": round(float(score) * 100, 2)}
            for name, score in ranked[:5]
        ]
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(getattr(model, "coef_"))[0]
        ranked = sorted(
            zip(FEATURE_NAMES, coefficients),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        info["top_importances"] = [
            {"label": name, "score": round(float(score), 4)}
            for name, score in ranked[:5]
        ]
    else:
        info["top_importances"] = []

    return info


class PlacementInput(BaseModel):
    CGPA: float = Field(ge=0, le=10)
    Internships: float = Field(ge=0)
    Projects: float = Field(ge=0)
    Certifications: float = Field(ge=0)
    Communication_Skills: float = Field(ge=0, le=100)
    Aptitude_Score: float = Field(ge=0, le=100)
    Backlogs: float = Field(ge=0)


try:
    model = load_model()
    model_runtime = "saved-model"
    # attempt to reconstruct training feature columns by dummy-encoding the bundled dataset
    try:
        _df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
        # drop obvious identifiers and target columns if present
        drop_cols = [c for c in ['student_id', 'StudentID', 'Placement', 'salary_package_lpa', 'package_lpa', 'company_type'] if c in _df.columns]
        _df_X = _df.drop(columns=drop_cols, errors='ignore')
        # fillna with empty string for categorical consistency
        _df_X = _df_X.fillna("")
        _dummy = pd.get_dummies(_df_X)
        n_expected = getattr(model, 'n_features_in_', _dummy.shape[1])
        model_feature_columns = list(_dummy.columns[:n_expected])
        # build defaults for missing columns from modes / zeros
        feature_defaults = {}
        for col in _df_X.columns:
            if _df_X[col].dtype == object:
                try:
                    feature_defaults[col] = _df_X[col].mode()[0]
                except Exception:
                    feature_defaults[col] = ""
            else:
                try:
                    feature_defaults[col] = float(_df_X[col].mean())
                except Exception:
                    feature_defaults[col] = 0.0
    except Exception:
        model_feature_columns = None
        feature_defaults = {}
except Exception:
    model = FallbackModel()
    model_runtime = "fallback"
insights = build_insights(load_rows())
model_info = build_model_info(model)
model_info["runtime"] = model_runtime
if model_runtime == "fallback":
    model_info["notes"].append("Fallback scoring is active because the saved model could not be loaded in the current environment.")


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "runtime": model_runtime}


@app.get("/api/insights")
def get_insights():
    return insights


@app.get("/api/project")
def get_project():
    return PROJECT_OVERVIEW


@app.get("/api/model-info")
def get_model_info():
    return model_info


@app.get("/api/site-data")
def get_site_data():
    return {
        "project": PROJECT_OVERVIEW,
        "model": model_info,
        "insights": insights,
    }


@app.post("/api/predict")
def predict(data: PlacementInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    # If the saved model expects more than the seven site-level inputs, build a matching vector
    if model_runtime == 'saved-model' and hasattr(model, 'n_features_in_') and getattr(model, 'n_features_in_', 0) != len(FEATURE_NAMES):
        try:
            # start from defaults reconstructed from the dataset
            base = {c: feature_defaults.get(c, 0.0) for c in feature_defaults}
            # overwrite with the seven provided inputs
            base.update({
                'CGPA': data.CGPA,
                'Internships': data.Internships,
                'Projects': data.Projects,
                'Certifications': data.Certifications,
                'Communication_Skills': data.Communication_Skills,
                'Aptitude_Score': data.Aptitude_Score,
                'Backlogs': data.Backlogs,
            })
            input_df = pd.DataFrame([base])
            input_dummy = pd.get_dummies(input_df).reindex(columns=model_feature_columns, fill_value=0)
            features = input_dummy.values.astype(float)
        except Exception:
            # fallback to the simple 7-feature vector if anything goes wrong
            features = np.array([
                data.CGPA,
                data.Internships,
                data.Projects,
                data.Certifications,
                data.Communication_Skills,
                data.Aptitude_Score,
                data.Backlogs,
            ], dtype=float).reshape(1, -1)
    else:
        features = np.array(
            [
                data.CGPA,
                data.Internships,
                data.Projects,
                data.Certifications,
                data.Communication_Skills,
                data.Aptitude_Score,
                data.Backlogs,
            ],
            dtype=float,
        ).reshape(1, -1)

    try:
        prediction = model.predict(features)[0]
    except ValueError as ve:
        msg = str(ve)
        # try to extract expected feature count from error message and pad
        import re

        m = re.search(r"expected:\s*(\d+),\s*got\s*(\d+)", msg)
        if m:
            expected = int(m.group(1))
            got = int(m.group(2))
            if got < expected:
                pad = np.zeros((1, expected - got), dtype=float)
                try:
                    padded = np.concatenate([features, pad], axis=1)
                    prediction = model.predict(padded)[0]
                    features = padded
                except Exception:
                    raise ve
            else:
                raise ve
        else:
            raise ve
    probability = None
    if hasattr(model, "predict_proba"):
        try:
            proba_values = model.predict_proba(features)[0]
            positive_index = 1
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                for candidate in (1, "1", True, "Placed", "placed", "Yes", "yes"):
                    if candidate in classes:
                        positive_index = classes.index(candidate)
                        break
            probability = float(proba_values[positive_index])
        except Exception:
            probability = None

    if isinstance(prediction, str):
        result_label = "Placed" if prediction.strip().lower() in {"1", "true", "yes", "placed"} else "Not Placed"
    else:
        result_label = "Placed" if int(prediction) == 1 else "Not Placed"

    return {
        "result": result_label,
        "probability": round(probability * 100, 2) if probability is not None else None,
        "readiness_score": readiness_score(data),
        "recommendation": role_recommendation(data),
        "next_steps": [
            "Strengthen projects and internships",
            "Improve aptitude and communication practice",
            "Keep academic backlog count at zero",
        ],
    }
