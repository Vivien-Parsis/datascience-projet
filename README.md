# 🔧 Système Intelligent de Maintenance Prédictive Industrielle

> Projet Data Science — EFREI M1 Dev. Manager Full Stack | 2025-26  
> Sujet 1 — Classification Binaire : prédiction de panne dans les 24h  
> 🔗 **Dépôt Git :** [github.com/Vivien-Parsis/datascience-projet](https://github.com/Vivien-Parsis/datascience-projet)

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
│  Raw Data  →  EDA  →  Preprocessing  →  Modélisation    │
│             (notebook)  (notebook)  (notebook)          │
│                              ↓                          │
│              artefacts/ (modèles .pkl, métriques .json) │
│                         ↙        ↘                      │
│              API FastAPI       Dashboard Streamlit      │
└─────────────────────────────────────────────────────────┘
```

---

## Dataset

**Source :** [Kaggle — Industrial Machine Predictive Maintenance](https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance)

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

📓 `EDA_Maintenance_Predictive.ipynb` — 35 cellules (12 code + 23 markdown)

- Audit complet : valeurs manquantes (2–4% sur 5 capteurs), outliers, distributions
- Analyse de corrélation : `temperature_motor` (+0.386) et `vibration_rms` (+0.264) sont les signaux les plus discriminants
- Vérification du déséquilibre de classes → stratégie de rééchantillonnage requise

### 2. Feature Engineering

📓 `data_preparation.ipynb` — 38 cellules (13 code + 25 markdown)

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
| **SMOTE ✅** | 50.0% | Génération synthétique (retenu) |
| Random OverSampling | 50.0% | Duplication aléatoire |
| Random UnderSampling | 50.0% | Réduction majoritaire |

`class_weight='balanced'` appliqué en complément sur les modèles compatibles.

### 5. Validation

- **StratifiedKFold(5)** — proportions de classes préservées dans chaque fold
- Métriques prioritaires : **Recall** (minimiser les pannes non détectées), F1-Score, ROC-AUC, PR-AUC

---

## Résultats des modèles

📓 `modeling.ipynb` — 44 cellules (16 code + 28 markdown)

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
- **Seuil production recommandé (0.05) :** Recall=1.000 — aucune panne non détectée, au prix de plus de fausses alertes. Adapté au contexte industriel où un faux négatif coûte bien plus cher qu'un faux positif.

---

## Installation

### Prérequis

- Python 3.10+
- pip
- Jupyter Notebook ou JupyterLab

### Dépendances

```bash
pip install -r requirements.txt
```

**`requirements.txt` :**

```none
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
notebook>=7.0
ipykernel>=6.0
```

---

## Lancement

> ⚠️ **Ordre d'exécution obligatoire** — les notebooks doivent être exécutés dans l'ordre ci-dessous : chacun produit les artefacts dont le suivant a besoin.

### Étape 1 — Analyse exploratoire

```bash
jupyter notebook EDA_Maintenance_Predictive.ipynb
```

Exécuter toutes les cellules (**Kernel > Restart & Run All**).  
Produit les figures dans `figures/eda/` (fig1 à fig7). Aucun artefact `.pkl` généré.

### Étape 2 — Préparation des données

```bash
jupyter notebook data_preparation.ipynb
```

Exécuter toutes les cellules. Génère dans `artefacts/` :
`preprocessor.pkl`, `X_train_proc.pkl`, `X_test_proc.pkl`, `y_train.pkl`, `y_test.pkl`, `feature_names.pkl`, `resampling_strategies.pkl`, `stratified_kfold.pkl`, `metadata.json`

### Étape 3 — Entraînement des modèles

```bash
jupyter notebook modeling.ipynb
```

> ⚠️ Prérequis : l'étape 2 doit avoir été exécutée en entier.

Exécuter toutes les cellules. Génère dans `artefacts/` :
`best_model.pkl`, `model_*.pkl`, `comparison_table.pkl`, `feature_importance.pkl`, `results.json`

### Étape 4 — Lancer l'API

```bash
cd app
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

> ⚠️ Prérequis : les étapes 2 et 3 doivent être complètes — l'API charge `preprocessor.pkl`, `best_model.pkl` et `results.json` au démarrage.

Documentation interactive (Swagger) : [http://localhost:8000/docs](http://localhost:8000/docs)

### Étape 5 — Lancer le Dashboard

```bash
cd app
streamlit run dashboard.py
```

> L'API (étape 4) doit être active pour que le simulateur fonctionne. Les autres pages fonctionnent en mode hors-ligne.

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
datascience-projet/
│
├── EDA_Maintenance_Predictive.ipynb       # Analyse exploratoire — 35 cellules (fig1 à fig7)
├── data_preparation.ipynb                 # Pipeline sklearn, split, SMOTE — 38 cellules (fig8)
├── modeling.ipynb                         # 4 modèles, évaluation, importance — 44 cellules (fig9 à fig13)
│
├── app/
│   ├── api.py                             # API FastAPI (predict, health, model-info, batch)
│   └── dashboard.py                       # Dashboard Streamlit (5 pages)
│
├── artefacts/                             # Générés par les notebooks (ne pas modifier manuellement)
│   ├── preprocessor.pkl                   # Pipeline sklearn fitté sur X_train
│   ├── best_model.pkl                     # Random Forest — modèle final déployé
│   ├── model_logistic_regression.pkl
│   ├── model_random_forest.pkl
│   ├── model_gradient_boosting.pkl
│   ├── model_mlp_deep_learning.pkl
│   ├── feature_names.pkl                  # Noms des features après OneHotEncoding
│   ├── feature_importance.pkl             # Permutation Importance DataFrame
│   ├── comparison_table.pkl               # Tableau comparatif des modèles
│   ├── results.json                       # Métriques et seuils de décision
│   └── metadata.json                      # Configuration du pipeline (pour l'API)
│
├── figures/
│   ├── eda/                               # Fig.1 à Fig.7 — générées par EDA_Maintenance_Predictive.ipynb
│   ├── data_preparation/                  # Fig.8 — générée par data_preparation.ipynb
│   └── models/                            # Fig.9 à Fig.13 — générées par modeling.ipynb
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
| Format d'analyse | Jupyter Notebooks | Documentation inline, cellules d'interprétation, reproductibilité |
| Imputation | Médiane | Robuste aux outliers des capteurs industriels |
| Encodage catégoriel | OneHotEncoder | Pas de relation ordinale entre types de machines |
| Normalisation | StandardScaler | Requis pour LR et MLP, neutre pour RF/XGB |
| Anti-leakage | `fit()` sur train uniquement | Garantit la validité des résultats sur le test set |
| Rééchantillonnage | SMOTE | Génère des exemples synthétiques réalistes vs duplication brute |
| Validation | StratifiedKFold(5) | Préserve les proportions de classes dans chaque fold |
| Métrique prioritaire | Recall | Minimiser les faux négatifs (pannes non détectées) |
| Modèle final | Random Forest | Meilleur compromis performance / vitesse / interprétabilité |
| Seuil production | 0.05 | Recall = 1.000 — aucune panne non détectée |
| API | FastAPI | Performance, validation Pydantic native, Swagger auto |
| Dashboard | Streamlit | Déploiement rapide, orienté métier |

---

## Auteur

- **Vivien PARSIS**
