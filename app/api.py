"""
=============================================================
API REST — Maintenance Prédictive Industrielle
FastAPI | Endpoints : /health  /predict  /model-info  /batch-predict
=============================================================
Lancement : uvicorn api:app --host 0.0.0.0 --port 8000 --reload
=============================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import joblib, json, numpy as np, pandas as pd
from datetime import datetime
import os

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES ARTEFACTS
# ─────────────────────────────────────────────────────────────
ARTDIR = "./artefacts"

try:
    preprocessor    = joblib.load(f"{ARTDIR}/preprocessor.pkl")
    best_model      = joblib.load(f"{ARTDIR}/best_model.pkl")
    feature_names   = joblib.load(f"{ARTDIR}/feature_names.pkl")
    feat_importance = joblib.load(f"{ARTDIR}/feature_importance.pkl")
    comparison_df   = joblib.load(f"{ARTDIR}/comparison_table.pkl")

    with open(f"{ARTDIR}/metadata.json") as f:
        metadata = json.load(f)
    with open(f"{ARTDIR}/results.json") as f:
        results = json.load(f)

    MODEL_LOADED = True
    LOAD_ERROR   = None
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR   = str(e)

ALL_FEATURES     = metadata["all_features"]      if MODEL_LOADED else []
NUMERIC_FEATURES = metadata["numeric_features"]  if MODEL_LOADED else []
CAT_FEATURES     = metadata["categorical_features"] if MODEL_LOADED else []
BEST_THRESHOLD   = results["recall_threshold"]   if MODEL_LOADED else 0.3

MACHINE_TYPES   = ["CNC", "Pump", "Compressor", "Robotic Arm"]
OPERATING_MODES = ["idle", "normal", "peak"]

# ─────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "🔧 Maintenance Prédictive — API",
    description = """
API de prédiction de pannes industrielles dans les 24 heures.

**Endpoints :**
- `GET  /health`         — état du service
- `POST /predict`        — prédiction unitaire
- `POST /batch-predict`  — prédictions en masse
- `GET  /model-info`     — infos sur le modèle déployé
- `GET  /features`       — liste des features attendues
    """,
    version = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# SCHÉMAS PYDANTIC
# ─────────────────────────────────────────────────────────────
class MachineInput(BaseModel):
    # Capteurs physiques
    vibration_rms          : float = Field(..., ge=0, le=15,   description="Vibration RMS (mm/s)", example=2.5)
    temperature_motor      : float = Field(..., ge=20, le=120, description="Température moteur (°C)", example=65.0)
    current_phase_avg      : Optional[float] = Field(None, ge=0, le=50, description="Courant moyen (A)", example=6.2)
    pressure_level         : Optional[float] = Field(None, ge=0, le=100, description="Pression (bar)", example=24.0)
    rpm                    : Optional[float] = Field(None, ge=0, le=5000, description="Vitesse de rotation (RPM)", example=1200.0)
    hours_since_maintenance: float = Field(..., ge=0, le=10000, description="Heures depuis dernière maintenance", example=300.0)
    ambient_temp           : float = Field(..., ge=-20, le=60, description="Température ambiante (°C)", example=22.0)
    rul_hours              : float = Field(..., ge=0, le=500,  description="Durée de vie restante estimée (h)", example=40.0)

    # Variables contextuelles
    machine_type  : str = Field(..., description="Type de machine", example="CNC")
    operating_mode: str = Field(..., description="Mode opératoire", example="normal")

    @validator('machine_type')
    def validate_machine_type(cls, v):
        if v not in MACHINE_TYPES:
            raise ValueError(f"machine_type doit être parmi : {MACHINE_TYPES}")
        return v

    @validator('operating_mode')
    def validate_operating_mode(cls, v):
        if v not in OPERATING_MODES:
            raise ValueError(f"operating_mode doit être parmi : {OPERATING_MODES}")
        return v

class PredictionResponse(BaseModel):
    prediction         : int
    label              : str
    probability_failure: float
    risk_level         : str
    threshold_used     : float
    top_risk_factors   : List[dict]
    recommendation     : str
    timestamp          : str

class BatchInput(BaseModel):
    machines: List[MachineInput]

class BatchResponse(BaseModel):
    total       : int
    at_risk     : int
    safe        : int
    predictions : List[PredictionResponse]

# ─────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────
def build_features(data: MachineInput) -> pd.DataFrame:
    """Construit le DataFrame avec feature engineering."""
    row = {feat: None for feat in ALL_FEATURES}
    row.update({
        'vibration_rms':           data.vibration_rms,
        'temperature_motor':       data.temperature_motor,
        'current_phase_avg':       data.current_phase_avg,
        'pressure_level':          data.pressure_level,
        'rpm':                     data.rpm,
        'hours_since_maintenance': data.hours_since_maintenance,
        'ambient_temp':            data.ambient_temp,
        'rul_hours':               data.rul_hours,
        'machine_type':            data.machine_type,
        'operating_mode':          data.operating_mode,
    })
    df = pd.DataFrame([row])

    # Feature engineering (identique au pipeline d'entraînement)
    df['temp_relative']      = df['temperature_motor'] - df['ambient_temp']
    df['vibration_per_rpm']  = df['vibration_rms'] / (df['rpm'].clip(lower=1) if df['rpm'].notna().all() else 1)
    df['maintenance_stress'] = np.log1p(df['hours_since_maintenance'])

    return df[ALL_FEATURES]

def get_risk_level(proba: float) -> str:
    if proba < 0.20:  return "🟢 Faible"
    if proba < 0.50:  return "🟡 Modéré"
    if proba < 0.75:  return "🟠 Élevé"
    return "🔴 Critique"

def get_recommendation(proba: float, label: int) -> str:
    if label == 0 and proba < 0.20:
        return "✅ Aucune action requise. Continuer la surveillance normale."
    if label == 0 and proba < 0.50:
        return "⚠️ Risque modéré. Planifier une inspection dans les 72h."
    if label == 1 and proba < 0.75:
        return "🔧 Panne probable dans 24h. Planifier une intervention préventive immédiate."
    return "🚨 Risque critique. Arrêt préventif recommandé. Intervention urgente requise."

def get_top_risk_factors(X_processed: np.ndarray) -> List[dict]:
    """Retourne les 5 features qui ont le plus influencé la prédiction."""
    imp = feat_importance.head(5)
    factors = []
    for _, row in imp.iterrows():
        feat_idx = list(feature_names).index(row['feature']) if row['feature'] in feature_names else -1
        val = float(X_processed[0, feat_idx]) if feat_idx >= 0 else 0.0
        factors.append({
            "feature":    row['feature'],
            "importance": round(float(row['importance']), 4),
            "value":      round(val, 3)
        })
    return factors

# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Vérifie que l'API et le modèle sont opérationnels."""
    return {
        "status"      : "ok" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "model_name"  : results.get("best_model_name", "N/A") if MODEL_LOADED else None,
        "error"       : LOAD_ERROR,
        "timestamp"   : datetime.now().isoformat(),
        "version"     : "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict(data: MachineInput):
    """
    Prédit si une machine va tomber en panne dans les 24 prochaines heures.

    **Retourne :**
    - `prediction` : 0 (pas de panne) ou 1 (panne probable)
    - `probability_failure` : probabilité de panne (0→1)
    - `risk_level` : 🟢 Faible | 🟡 Modéré | 🟠 Élevé | 🔴 Critique
    - `top_risk_factors` : variables les plus influentes
    - `recommendation` : action recommandée
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail=f"Modèle non chargé : {LOAD_ERROR}")

    try:
        df_input   = build_features(data)
        X_proc     = preprocessor.transform(df_input)
        proba      = float(best_model.predict_proba(X_proc)[0, 1])
        prediction = int(proba >= BEST_THRESHOLD)

        return PredictionResponse(
            prediction          = prediction,
            label               = "Panne probable dans 24h" if prediction == 1 else "Pas de panne détectée",
            probability_failure = round(proba, 4),
            risk_level          = get_risk_level(proba),
            threshold_used      = BEST_THRESHOLD,
            top_risk_factors    = get_top_risk_factors(X_proc),
            recommendation      = get_recommendation(proba, prediction),
            timestamp           = datetime.now().isoformat()
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")


@app.post("/batch-predict", response_model=BatchResponse, tags=["Prédiction"])
def batch_predict(data: BatchInput):
    """Prédictions en masse pour plusieurs machines simultanément."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    if len(data.machines) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 machines par requête")

    predictions = []
    for machine in data.machines:
        df_input = build_features(machine)
        X_proc   = preprocessor.transform(df_input)
        proba    = float(best_model.predict_proba(X_proc)[0, 1])
        pred     = int(proba >= BEST_THRESHOLD)
        predictions.append(PredictionResponse(
            prediction          = pred,
            label               = "Panne probable dans 24h" if pred == 1 else "Pas de panne",
            probability_failure = round(proba, 4),
            risk_level          = get_risk_level(proba),
            threshold_used      = BEST_THRESHOLD,
            top_risk_factors    = get_top_risk_factors(X_proc),
            recommendation      = get_recommendation(proba, pred),
            timestamp           = datetime.now().isoformat()
        ))

    at_risk = sum(p.prediction for p in predictions)
    return BatchResponse(
        total       = len(predictions),
        at_risk     = at_risk,
        safe        = len(predictions) - at_risk,
        predictions = predictions
    )


@app.get("/model-info", tags=["Monitoring"])
def model_info():
    """Informations détaillées sur le modèle déployé."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    return {
        "model_name"          : results["best_model_name"],
        "version"             : "1.0.0",
        "task"                : "Binary Classification — failure_within_24h",
        "threshold"           : BEST_THRESHOLD,
        "n_features"          : len(ALL_FEATURES),
        "numeric_features"    : NUMERIC_FEATURES,
        "categorical_features": CAT_FEATURES,
        "training_info"       : {
            "n_train"       : metadata["n_train"],
            "n_test"        : metadata["n_test"],
            "class_balance" : metadata["class_balance"],
            "resampling"    : "SMOTE"
        },
        "test_metrics"        : results["test_metrics"][results["best_model_name"]],
        "top_features"        : feat_importance.head(5)[["feature","importance"]].to_dict("records"),
        "valid_machine_types" : MACHINE_TYPES,
        "valid_operating_modes": OPERATING_MODES
    }


@app.get("/features", tags=["Monitoring"])
def get_features():
    """Liste des features attendues avec leurs contraintes."""
    return {
        "required_fields": [
            {"name": "vibration_rms",           "type": "float", "range": "0–15",     "unit": "mm/s"},
            {"name": "temperature_motor",        "type": "float", "range": "20–120",   "unit": "°C"},
            {"name": "hours_since_maintenance",  "type": "float", "range": "0–10000",  "unit": "heures"},
            {"name": "ambient_temp",             "type": "float", "range": "-20–60",   "unit": "°C"},
            {"name": "rul_hours",                "type": "float", "range": "0–500",    "unit": "heures"},
            {"name": "machine_type",             "type": "str",   "values": MACHINE_TYPES},
            {"name": "operating_mode",           "type": "str",   "values": OPERATING_MODES},
        ],
        "optional_fields": [
            {"name": "current_phase_avg", "type": "float", "range": "0–50",   "unit": "A",   "default": "médiane"},
            {"name": "pressure_level",    "type": "float", "range": "0–100",  "unit": "bar", "default": "médiane"},
            {"name": "rpm",               "type": "float", "range": "0–5000", "unit": "RPM", "default": "médiane"},
        ],
        "example_request": {
            "vibration_rms": 3.5, "temperature_motor": 72.0,
            "current_phase_avg": 7.1, "pressure_level": 26.0, "rpm": 1450.0,
            "hours_since_maintenance": 380.0, "ambient_temp": 21.0,
            "rul_hours": 18.0, "machine_type": "Compressor", "operating_mode": "peak"
        }
    }
