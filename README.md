# 🔧 Système Intelligent de Maintenance Prédictive Industrielle

> Projet Data Science — EFREI M1 Dev. Manager Full Stack | 2025-26  
> Sujet 1 — Classification Binaire : prédiction de panne dans les 24h

---

## 📋 Table des matières

- [Contexte et objectif](#contexte-et-objectif)
- [Architecture du projet](#architecture-du-projet)
- [Dataset](#dataset)
- [Pipeline Data Science](#pipeline-data-science)
- [Résultats des modèles](#résultats-des-modèles)
- [Installation](#installation)
- [Lancement](#lancement)
- [API REST](#api-rest)
- [Dashboard](#dashboard)
- [Structure des fichiers](#structure-des-fichiers)
- [Choix techniques](#choix-techniques)

---

## Contexte et objectif

Ce projet conçoit un **système intelligent de maintenance prédictive** capable d'anticiper les défaillances d'équipements industriels à partir de données capteurs.

**Tâche prédictive choisie :** Classification binaire — prédire si une machine va tomber en panne dans les **24 prochaines heures**.

**Variable cible :** `failure_within_24h` (0 = pas de panne, 1 = panne imminente)

**Valeur métier :**

- Réduire les arrêts non planifiés et les coûts de maintenance corrective
- Prioriser les interventions préventives
- Améliorer la disponibilité des machines (uptime)

---

## Architecture du projet

```none
┌─────────────────────────────────────────────────────────┐
│                     PIPELINE COMPLET                    │
│                                                         │
│  Raw Data  →  EDA  →  Preprocessing  →  Modélisation   │
│                              ↓                          │
│              artefacts/ (modèles .pkl, métriques .json) │
│                         ↙        ↘                      │
│              API FastAPI       Dashboard Streamlit       │
└─────────────────────────────────────────────────────────┘
```

---

## Dataset

| Caractéristique | Valeur |
| --- | --- |
| Enregistrements | 24 042 |
| Variables | 15 |
| Machines | 20 (4 types) |
| Période | Janvier 2024 |
| Déséquilibre des classes | 85.2% / 14.8% (ratio ~5.75:1) |

**Variables capteurs utilisées :**

| Variable | Type | Description |
| --- | --- | --- |
| `vibration_rms` | Numérique | Vibration RMS (mm/s) |
| `temperature_motor` | Numérique | Température moteur (°C) |
| `current_phase_avg` | Numérique | Courant moyen (A) |
| `pressure_level` | Numérique | Pression (bar) |
| `rpm` | Numérique | Vitesse de rotation |
| `hours_since_maintenance` | Numérique | Heures depuis la dernière maintenance |
| `ambient_temp` | Numérique | Température ambiante (°C) |
| `rul_hours` | Numérique | Durée de vie restante estimée (h) |
| `machine_type` | Catégoriel | CNC / Pump / Compressor / Robotic Arm |
| `operating_mode` | Catégoriel | idle / normal / peak |

**Colonnes exclues (data leakage) :**

- `failure_type` — révèle directement la cible (100% corrélé)
- `estimated_repair_cost` — toujours 0 si pas de panne
- `timestamp`, `machine_id` — identifiants sans signal prédictif

---

## Pipeline Data Science

### 1. Analyse exploratoire (EDA)

- Audit complet : valeurs manquantes (2–4% sur 5 capteurs), outliers, distributions
- Analyse de corrélation : `temperature_motor` (+0.386) et `vibration_rms` (+0.264) sont les signaux les plus discriminants
- Vérification du déséquilibre de classes → stratégie de rééchantillonnage requise

### 2. Feature Engineering

Trois nouvelles variables créées :

| Feature | Formule | Corrélation cible |
| --- | --- | --- |
| `temp_relative` | `temperature_motor − ambient_temp` | +0.375 |
| `vibration_per_rpm` | `vibration_rms / rpm` | +0.317 |
| `maintenance_stress` | `log(1 + hours_since_maintenance)` | +0.075 |

### 3. Pipeline sklearn

```none
Numérique  : SimpleImputer(median) → StandardScaler
Catégoriel : SimpleImputer(mode)   → OneHotEncoder
→ Fit UNIQUEMENT sur X_train (pas de data leakage)
```

- Split stratifié 80/20 : **19 233 train** / **4 809 test**
- Proportions de classes préservées : 14.8% de positifs dans les deux sets

### 4. Gestion du déséquilibre

Quatre stratégies comparées :

| Stratégie | Positifs train | Approche |
| --- | --- | --- |
| Baseline | 14.8% | Aucune |
| SMOTE ✅ | 50.0% | Génération synthétique (retenu) |
| Random OverSampling | 50.0% | Duplication aléatoire |
| Random UnderSampling | 50.0% | Réduction majoritaire |

`class_weight='balanced'` appliqué en complément sur les modèles compatibles.

### 5. Validation

- **StratifiedKFold(5)** — proportions de classes préservées dans chaque fold
- Métriques prioritaires : **Recall** (minimiser les pannes non détectées), F1-Score, ROC-AUC, PR-AUC

---

## Résultats des modèles

Évaluation sur le **jeu de test réel** (4 809 observations, non rééchantillonné) :

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Temps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.929 | 0.700 | 0.909 | 0.791 | 0.972 | 0.891 | 0.1s |
| **Random Forest** ⭐ | **0.998** | **0.992** | **0.993** | **0.992** | **1.000** | **1.000** | 9.7s |
| Gradient Boosting | 0.998 | 0.990 | 0.993 | 0.992 | 1.000 | 0.999 | 30.2s |
| MLP (Deep Learning) | 0.991 | 0.971 | 0.972 | 0.971 | 0.999 | 0.995 | 9.4s |

### Modèle sélectionné : Random Forest

**Justification :**

- Performances identiques au Gradient Boosting mais **3× plus rapide** à entraîner (9.7s vs 30.2s)
- Feature importance native — interprétable pour un responsable maintenance
- Stabilité cross-validation : CV-F1 = 0.998 ± 0.001
- Déploiement plus simple et reproductible

**Le MLP n'est pas sélectionné** malgré ses bonnes performances car il n'apporte aucun gain mesurable sur ce jeu de données tabulaire structuré, pour une complexité et une interprétabilité bien inférieures.

### Top 5 variables les plus importantes (Permutation Importance)

| Rang | Feature | Importance | Interprétation |
| --- | --- | --- | --- |
| 1 | `rul_hours` | 0.495 | Signal dominant — durée de vie restante |
| 2 | `vibration_per_rpm` *(engineered)* | 0.095 | Vibration anormale relative à la vitesse |
| 3 | `rpm` | 0.047 | Variations de vitesse corrélées aux pannes |
| 4 | `temperature_motor` | 0.040 | Surchauffe = signe de défaillance |
| 5 | `current_phase_avg` | 0.013 | Surconsommation électrique |

### Seuil de décision

- **Seuil défaut (0.5) :** F1=0.992, Recall=0.993
- **Seuil production recommandé (0.05) :** Recall=1.000 — aucune panne non détectée, au prix de plus de fausses alertes. Adapté au contexte industriel où un faux négatif (panne non détectée) coûte bien plus cher qu'un faux positif (fausse alerte).

---

## Installation

### Prérequis

- Python 3.10+
- pip

### Dépendances

```bash
pip install -r requirements.txt
```

**`requirements.txt` :**

```bash
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
imbalanced-learn>=0.11
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15
fastapi>=0.104
uvicorn>=0.24
streamlit>=1.28
pydantic>=2.0
joblib>=1.3
requests>=2.31
```

---

## Lancement

### Étape 1 — Préparation des données

```bash
python data_preparation.py
```

Génère tous les artefacts dans `artefacts/` : preprocessor, splits train/test, stratégies de rééchantillonnage.

### Étape 2 — Entraînement des modèles

```bash
python modeling.py
```

Entraîne les 4 modèles, génère les figures de comparaison et sauvegarde `best_model.pkl`.

### Étape 3 — Lancer l'API

```bash
cd app
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Documentation interactive (Swagger) : [http://localhost:8000/docs](http://localhost:8000/docs)

### Étape 4 — Lancer le Dashboard

```bash
cd app
streamlit run dashboard.py
```

Dashboard accessible à : [http://localhost:8501](http://localhost:8501)

---

## API REST

### `GET /health`

Vérifie l'état du service.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "Random Forest",
  "version": "1.0.0"
}
```

---

### `POST /predict`

Prédit la probabilité de panne pour une machine.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vibration_rms": 3.5,
    "temperature_motor": 72.0,
    "current_phase_avg": 7.1,
    "pressure_level": 26.0,
    "rpm": 1450.0,
    "hours_since_maintenance": 380.0,
    "ambient_temp": 21.0,
    "rul_hours": 18.0,
    "machine_type": "Compressor",
    "operating_mode": "peak"
  }'
```

**Réponse :**

```json
{
  "prediction": 1,
  "label": "Panne probable dans 24h",
  "probability_failure": 0.9187,
  "risk_level": "🔴 Critique",
  "threshold_used": 0.05,
  "top_risk_factors": [
    {"feature": "rul_hours", "importance": 0.4955, "value": -0.372},
    {"feature": "vibration_per_rpm", "importance": 0.0951, "value": 0.587}
  ],
  "recommendation": "🚨 Risque critique. Arrêt préventif recommandé. Intervention urgente requise.",
  "timestamp": "2026-01-01T10:00:00"
}
```

**Champs requis :** `vibration_rms`, `temperature_motor`, `hours_since_maintenance`, `ambient_temp`, `rul_hours`, `machine_type`, `operating_mode`

**Champs optionnels** (imputés par médiane si absents) : `current_phase_avg`, `pressure_level`, `rpm`

---

### `POST /batch-predict`

Prédictions en masse (jusqu'à 100 machines).

```json
{
  "machines": [ { ...machine1... }, { ...machine2... } ]
}
```

---

### `GET /model-info`

Retourne les métriques, features et configuration du modèle déployé.

---

### `GET /features`

Liste toutes les features attendues avec leurs types, plages et valeurs valides.

---

## Dashboard

5 pages disponibles :

| Page | Contenu |
| --- | --- |
| **Vue d'ensemble** | KPIs globaux, taux de panne par machine et mode, distributions des capteurs |
| **Simulateur** | Saisie des paramètres → prédiction temps réel via l'API, jauge de risque, recommandation |
| **Comparaison des modèles** | Radar chart, tableau comparatif, justification du choix |
| **Importance des variables** | Permutation Importance interactive, interprétation métier |
| **API & Infos système** | Statut API, exemple de requête, architecture |

---

## Structure des fichiers

```none
projet_maintenance/
│
├── data_preparation.py        # Pipeline sklearn : nettoyage, features, split, rééchantillonnage
├── modeling.py                # Entraînement 4 modèles, évaluation, feature importance
├── eda_maintenance.py         # Analyse exploratoire complète (7 figures)
│
├── app/
│   ├── api.py                 # API FastAPI (predict, health, model-info, batch)
│   └── dashboard.py           # Dashboard Streamlit (5 pages)
│
├── artefacts/
│   ├── preprocessor.pkl       # Pipeline sklearn fitté sur X_train
│   ├── best_model.pkl         # Random Forest — modèle final
│   ├── model_*.pkl            # Tous les modèles entraînés
│   ├── feature_names.pkl      # Noms des features après OneHotEncoding
│   ├── feature_importance.pkl # Permutation Importance DataFrame
│   ├── comparison_table.pkl   # Tableau comparatif des modèles
│   ├── results.json           # Métriques et seuils de décision
│   └── metadata.json          # Configuration du pipeline
│
├── figures/
│   ├── /eda                   # Figures EDA (fig1 à fig7)
│   ├── /data_preparation      # Figures préparation (fig8)
│   └── /models               # Figures modélisation (fig9 à fig13)
│
├── industrial_machine_maintenance.csv
├── requirements.txt
└── README.md
```

---

## Choix techniques

| Décision | Choix | Justification |
| --- | --- | --- |
| Tâche | Classification binaire | `failure_within_24h` — tâche la plus actionnelle |
| Imputation | Médiane | Robuste aux outliers des capteurs |
| Encodage catégoriel | OneHotEncoder | Pas de relation ordinale entre types de machines |
| Normalisation | StandardScaler | Requis pour LR et MLP, neutre pour RF/XGB |
| Anti-leakage | `fit()` sur train uniquement | Garantit la validité des résultats |
| Rééchantillonnage | SMOTE | Génère des exemples synthétiques réalistes vs duplication brute |
| Validation | StratifiedKFold(5) | Préserve les proportions de classes dans chaque fold |
| Métrique prioritaire | Recall | Minimiser les faux négatifs (pannes non détectées) |
| Modèle final | Random Forest | Meilleur compromis performance / vitesse / interprétabilité |
| Seuil production | 0.05 | Recall = 1.000 — aucune panne non détectée |
| API | FastAPI | Performance, validation Pydantic native, Swagger auto |
| Dashboard | Streamlit | Déploiement rapide, orienté métier |

---

## Auteurs

- Vivien PARSIS
