"""
=============================================================
ÉTAPE 2 — Préparation des données & Pipeline sklearn
Projet : Maintenance Prédictive Industrielle
Tâche   : failure_within_24h (Classification Binaire)
=============================================================
Contenu :
  1. Nettoyage & audit
  2. Feature engineering
  3. Détection data leakage
  4. Pipeline sklearn (imputation → encodage → normalisation)
  5. Split stratifié train/test
  6. Gestion déséquilibre (baseline, class_weight, SMOTE)
  7. Sauvegarde des artefacts
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os, joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

sns.set_theme(style="whitegrid", palette="Set2")
COLORS = {"no_fail": "#4C9BE8", "fail": "#E85D5D"}
TARGET  = "failure_within_24h"
OUTDIR  = "./outputs"
ARTDIR  = "./artefacts"
FIGDIR  = "./figures/data_preparation"
for d in [ARTDIR, FIGDIR]: os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. CHARGEMENT & AUDIT INITIAL
# ─────────────────────────────────────────────────────────────
print("="*60)
print("1. CHARGEMENT & AUDIT")
print("="*60)

df = pd.read_csv('./industrial_machine_maintenance.csv',
                 parse_dates=['timestamp'])
print(f"Shape initiale : {df.shape}")

# ─────────────────────────────────────────────────────────────
# 2. DÉTECTION DATA LEAKAGE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. DÉTECTION DATA LEAKAGE")
print("="*60)

"""
VARIABLES À EXCLURE & RAISONS :
  - timestamp   : identifiant temporel, pas un capteur
  - machine_id  : identifiant, pas un signal physique
  - failure_type: RÉVÈLE LA CIBLE (si failure_type != 'none' → panne = 1)
                  → LEAKAGE CRITIQUE
  - rul_hours   : Remaining Useful Life — quasi proxy de la cible
                  (fortement corrélé, parfois calculé à partir de failure_within_24h)
                  → À CONSERVER AVEC PRUDENCE (on l'inclut mais on surveille)
  - estimated_repair_cost: 0 si pas de panne, >0 si panne → LEAKAGE
"""

leakage_check = pd.crosstab(df['failure_type'], df[TARGET])
print("failure_type vs failure_within_24h (leakage évident) :")
print(leakage_check)
print("\n→ failure_type='none' ↔ target=0 TOUJOURS : LEAKAGE CRITIQUE")
print("→ estimated_repair_cost=0 ↔ target=0 TOUJOURS :", (df[df[TARGET]==0]['estimated_repair_cost']==0).all())

COLS_DROP = ['timestamp', 'machine_id', 'failure_type', 'estimated_repair_cost']
print(f"\nColonnes supprimées (leakage/identifiants): {COLS_DROP}")

# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. FEATURE ENGINEERING")
print("="*60)

df_clean = df.drop(columns=COLS_DROP).copy()

# Feature 1 : Température relative (écart à la température ambiante)
# → Capte la surchauffe moteur indépendamment de la saison
df_clean['temp_relative'] = df_clean['temperature_motor'] - df_clean['ambient_temp']
print("✅ temp_relative = temperature_motor - ambient_temp")
print(f"   Corrélation avec cible : {df_clean['temp_relative'].corr(df_clean[TARGET]):.3f}")

# Feature 2 : Ratio vibration / rpm (vibration normalisée par la vitesse)
# → Detect anomalies vibratoires indépendantes de la vitesse
df_clean['vibration_per_rpm'] = df_clean['vibration_rms'] / (df_clean['rpm'].clip(lower=1))
print("✅ vibration_per_rpm = vibration_rms / rpm")
print(f"   Corrélation avec cible : {df_clean['vibration_per_rpm'].corr(df_clean[TARGET]):.3f}")

# Feature 3 : Score de stress cumulé (maintenance overdue)
# → Plus la machine tourne longtemps sans maintenance, plus le risque monte
df_clean['maintenance_stress'] = np.log1p(df_clean['hours_since_maintenance'])
print("✅ maintenance_stress = log(1 + hours_since_maintenance)")
print(f"   Corrélation avec cible : {df_clean['maintenance_stress'].corr(df_clean[TARGET]):.3f}")

print(f"\nShape après feature engineering : {df_clean.shape}")

# ─────────────────────────────────────────────────────────────
# 4. DÉFINITION DES FEATURES
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. DÉFINITION DES FEATURES")
print("="*60)

NUMERIC_FEATURES = [
    'vibration_rms', 'temperature_motor', 'current_phase_avg',
    'pressure_level', 'rpm', 'hours_since_maintenance', 'ambient_temp',
    'rul_hours',
    # Features engineered
    'temp_relative', 'vibration_per_rpm', 'maintenance_stress'
]
CATEGORICAL_FEATURES = ['machine_type', 'operating_mode']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

print(f"Features numériques ({len(NUMERIC_FEATURES)}) : {NUMERIC_FEATURES}")
print(f"Features catégorielles ({len(CATEGORICAL_FEATURES)}) : {CATEGORICAL_FEATURES}")
print(f"Total features : {len(ALL_FEATURES)}")

X = df_clean[ALL_FEATURES]
y = df_clean[TARGET]

# ─────────────────────────────────────────────────────────────
# 5. SPLIT STRATIFIÉ TRAIN / TEST
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. SPLIT STRATIFIÉ (80/20)")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train : {X_train.shape[0]:,} obs  | Test : {X_test.shape[0]:,} obs")
print(f"Train positifs : {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"Test  positifs : {y_test.sum():,}  ({y_test.mean()*100:.1f}%)")
print("→ Stratified split : proportions préservées ✅")

# ─────────────────────────────────────────────────────────────
# 6. PIPELINE SKLEARN (sans data leakage)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("6. CONSTRUCTION DU PIPELINE SKLEARN")
print("="*60)

# Sous-pipeline numérique
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

# Sous-pipeline catégoriel
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Assemblage ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, NUMERIC_FEATURES),
    ('cat', categorical_transformer, CATEGORICAL_FEATURES)
])

print("Pipeline sklearn construit :")
print("  Numérique  : SimpleImputer(median) → StandardScaler")
print("  Catégoriel : SimpleImputer(mode)   → OneHotEncoder")
print("→ FIT uniquement sur X_train (pas de data leakage) ✅")

# Fit sur train uniquement, transform sur train ET test
preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Noms des features après OneHot
cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(CATEGORICAL_FEATURES)
feature_names_out = NUMERIC_FEATURES + list(cat_feature_names)

print(f"\nShape X_train après preprocessing : {X_train_proc.shape}")
print(f"Shape X_test  après preprocessing : {X_test_proc.shape}")
print(f"Features finales ({len(feature_names_out)}) : {feature_names_out}")

# ─────────────────────────────────────────────────────────────
# 7. GESTION DU DÉSÉQUILIBRE — Comparaison des stratégies
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("7. STRATÉGIES DE GESTION DU DÉSÉQUILIBRE")
print("="*60)

strategies = {}

# 7a — Baseline (aucun rééchantillonnage)
strategies['baseline'] = (X_train_proc, y_train.values)
print(f"Baseline     : {y_train.sum():,} positifs / {len(y_train):,} total ({y_train.mean()*100:.1f}%)")

# 7b — Random Over-Sampling
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train_proc, y_train)
strategies['over_sampling'] = (X_ros, y_ros)
print(f"OverSampling : {y_ros.sum():,} positifs / {len(y_ros):,} total ({y_ros.mean()*100:.1f}%)")

# 7c — SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X_train_proc, y_train)
strategies['smote'] = (X_smote, y_smote)
print(f"SMOTE        : {y_smote.sum():,} positifs / {len(y_smote):,} total ({y_smote.mean()*100:.1f}%)")

# 7d — Random Under-Sampling
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train_proc, y_train)
strategies['under_sampling'] = (X_rus, y_rus)
print(f"UnderSampling: {y_rus.sum():,} positifs / {len(y_rus):,} total ({y_rus.mean()*100:.1f}%)")

print("\n→ Stratégie recommandée pour la modélisation : SMOTE + class_weight")
print("→ class_weight='balanced' testé sur chaque modèle en complément")

# ─────────────────────────────────────────────────────────────
# 8. CROSS-VALIDATION STRATIFIÉE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("8. CROSS-VALIDATION STRATIFIÉE")
print("="*60)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("StratifiedKFold(n_splits=5) — proportions de classes préservées ✅")
print("→ Sera utilisée lors de l'entraînement des modèles")
for i, (_, val_idx) in enumerate(skf.split(X_train_proc, y_train)):
    fold_pos = y_train.values[val_idx].mean() * 100
    print(f"  Fold {i+1} : {len(val_idx):,} obs | {fold_pos:.1f}% positifs")

# ─────────────────────────────────────────────────────────────
# 9. VISUALISATIONS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("9. GÉNÉRATION DES FIGURES DE PRÉPARATION")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Fig.8 — Pipeline de préparation des données", fontsize=15, fontweight='bold')

# 8a — Stratégies de rééchantillonnage
ax = axes[0, 0]
labels  = ['Baseline', 'OverSampling', 'SMOTE', 'UnderSampling']
n_total = [len(v[1]) for v in strategies.values()]
n_pos   = [v[1].sum() for v in strategies.values()]
n_neg   = [t - p for t, p in zip(n_total, n_pos)]
x = np.arange(len(labels))
ax.bar(x, n_neg, label='Classe 0 (pas de panne)', color=COLORS['no_fail'], alpha=0.8)
ax.bar(x, n_pos, bottom=n_neg, label='Classe 1 (panne)', color=COLORS['fail'], alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_title("Stratégies rééchantillonnage\n(taille du train set)", fontweight='bold')
ax.set_ylabel("Nb observations")
ax.legend(fontsize=8)
for i, (t, p) in enumerate(zip(n_total, n_pos)):
    ax.text(i, t + 200, f'{p/t*100:.0f}%\npositifs', ha='center', fontsize=8, fontweight='bold', color=COLORS['fail'])

# 8b — Distribution avant/après StandardScaler (vibration_rms)
ax = axes[0, 1]
col_idx = NUMERIC_FEATURES.index('vibration_rms')
raw_train = X_train['vibration_rms'].dropna()
scaled_train = X_train_proc[:, col_idx]
ax.hist(raw_train, bins=40, alpha=0.6, color='#94C5F8', density=True, label='Brut')
ax.hist(scaled_train, bins=40, alpha=0.6, color='#F4A261', density=True, label='Normalisé (StandardScaler)')
ax.set_title("vibration_rms\nAvant / Après StandardScaler", fontweight='bold')
ax.legend(); ax.set_xlabel("Valeur"); ax.set_ylabel("Densité")

# 8c — Distribution avant/après (temperature_motor)
ax = axes[0, 2]
col_idx2 = NUMERIC_FEATURES.index('temperature_motor')
raw2 = X_train['temperature_motor'].dropna()
scaled2 = X_train_proc[:, col_idx2]
ax.hist(raw2, bins=40, alpha=0.6, color='#94C5F8', density=True, label='Brut')
ax.hist(scaled2, bins=40, alpha=0.6, color='#F4A261', density=True, label='Normalisé')
ax.set_title("temperature_motor\nAvant / Après StandardScaler", fontweight='bold')
ax.legend(); ax.set_xlabel("Valeur"); ax.set_ylabel("Densité")

# 8d — Corrélations features engineered vs originales
ax = axes[1, 0]
feat_corr = df_clean[NUMERIC_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
colors_bar = [COLORS['fail'] if v > 0 else COLORS['no_fail'] for v in feat_corr.values]
bars = ax.barh(feat_corr.index, feat_corr.values, color=colors_bar, alpha=0.85, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
# Surligner les features engineered
for i, name in enumerate(feat_corr.index):
    if name in ['temp_relative', 'vibration_per_rpm', 'maintenance_stress']:
        ax.get_yticklabels()[i].set_color('#E8700A')
        ax.get_yticklabels()[i].set_fontweight('bold')
for bar, val in zip(bars, feat_corr.values):
    ax.text(val + (0.003 if val >= 0 else -0.003), bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=7.5)
ax.set_title("Corrélations avec la cible\n(orange = features engineered)", fontweight='bold')
ax.set_xlabel("Corrélation de Pearson")

# 8e — Split train/test (proportions)
ax = axes[1, 1]
split_data = {
    'Train\n(19 233)': [y_train.value_counts()[0], y_train.value_counts()[1]],
    'Test\n(4 809)':   [y_test.value_counts()[0],  y_test.value_counts()[1]]
}
x = np.arange(2)
bot0 = [split_data['Train\n(19 233)'][0], split_data['Test\n(4 809)'][0]]
bot1 = [split_data['Train\n(19 233)'][1], split_data['Test\n(4 809)'][1]]
ax.bar(x, bot0, color=COLORS['no_fail'], alpha=0.8, label='Classe 0')
ax.bar(x, bot1, bottom=bot0, color=COLORS['fail'], alpha=0.8, label='Classe 1')
ax.set_xticks(x)
ax.set_xticklabels(['Train\n(19 233)', 'Test\n(4 809)'])
for xi, (t, p) in enumerate(zip([len(y_train), len(y_test)], [y_train.sum(), y_test.sum()])):
    ax.text(xi, t + 100, f'{p/t*100:.1f}%\npositifs', ha='center', fontsize=9, fontweight='bold', color=COLORS['fail'])
ax.set_title("Split stratifié train/test (80/20)\n→ Proportions préservées", fontweight='bold')
ax.set_ylabel("Nb observations"); ax.legend(fontsize=8)

# 8f — Schéma du pipeline
ax = axes[1, 2]
ax.axis('off')
pipeline_text = """
   PIPELINE DE PRÉPARATION
   ========================

   RAW DATA (24 042 obs)
          │
   ┌──────▼──────────────────┐
   │  DROP colonnes leakage  │
   │  failure_type           │
   │  estimated_repair_cost  │
   │  timestamp, machine_id  │
   └──────┬──────────────────┘
          │
   ┌──────▼──────────────────┐
   │  FEATURE ENGINEERING    │
   │  temp_relative          │
   │  vibration_per_rpm      │
   │  maintenance_stress     │
   └──────┬──────────────────┘
          │
   ┌──────▼──────────────────┐
   │  STRATIFIED SPLIT 80/20 │
   │  Train: 19 233 obs      │
   │  Test : 4 809 obs       │
   └──────┬──────────────────┘
          │ (fit sur TRAIN uniquement)
   ┌──────▼──────────────────┐
   │  COLUMN TRANSFORMER     │
   │  Num → Impute + Scale   │
   │  Cat → Impute + OHE     │
   └──────┬──────────────────┘
          │
   ┌──────▼──────────────────┐
   │  RÉÉCHANTILLONNAGE      │
   │  SMOTE / class_weight   │
   └──────┬──────────────────┘
          │
   ┌──────▼──────────────────┐
   │  MODÉLISATION (étape 3) │
   └─────────────────────────┘
"""
ax.text(0.05, 0.95, pipeline_text, transform=ax.transAxes,
        fontsize=8.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.8))

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig8_preparation.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig.8 générée")

# ─────────────────────────────────────────────────────────────
# 10. SAUVEGARDE DES ARTEFACTS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("10. SAUVEGARDE DES ARTEFACTS")
print("="*60)

# Préprocesseur (pour le réutiliser dans les modèles et l'API)
joblib.dump(preprocessor,       f"{ARTDIR}/preprocessor.pkl")
joblib.dump(X_train_proc,       f"{ARTDIR}/X_train_proc.pkl")
joblib.dump(X_test_proc,        f"{ARTDIR}/X_test_proc.pkl")
joblib.dump(y_train.values,     f"{ARTDIR}/y_train.pkl")
joblib.dump(y_test.values,      f"{ARTDIR}/y_test.pkl")
joblib.dump(feature_names_out,  f"{ARTDIR}/feature_names.pkl")
joblib.dump(strategies,         f"{ARTDIR}/resampling_strategies.pkl")
joblib.dump(skf,                f"{ARTDIR}/stratified_kfold.pkl")

# Métadonnées (pour l'API)
import json
meta = {
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "all_features": ALL_FEATURES,
    "target": TARGET,
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "class_balance": {"0": int(y_train.value_counts()[0]), "1": int(y_train.value_counts()[1])},
    "dropped_columns": COLS_DROP,
    "engineered_features": ["temp_relative", "vibration_per_rpm", "maintenance_stress"]
}
with open(f"{ARTDIR}/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Artefacts sauvegardés :")
for fname in os.listdir(ARTDIR):
    size = os.path.getsize(f"{ARTDIR}/{fname}")
    print(f"  {fname:<35} {size:>10,} bytes")

print("\n" + "="*60)
print("✅ PRÉPARATION TERMINÉE — Prêt pour la modélisation")
print("="*60)
print(f"""
RÉSUMÉ :
  • {len(ALL_FEATURES)} features en entrée → {X_train_proc.shape[1]} features après OHE
  • Train : {X_train_proc.shape[0]:,} obs | Test : {X_test_proc.shape[0]:,} obs
  • Stratégies disponibles : baseline, over_sampling, smote, under_sampling
  • Cross-validation : StratifiedKFold(5)
  • Artefacts sauvegardés dans : {ARTDIR}
""")
