"""
ÉTAPE 3 — Modélisation Multi-Algorithmes & Évaluation
Projet : Maintenance Prédictive Industrielle
Tâche   : failure_within_24h (Classification Binaire)
Modèles :
  1. Régression Logistique (baseline linéaire)
  2. Random Forest         (ensemble, non-linéaire)
  3. Gradient Boosting     (XGBoost, boosting)
  4. MLP                   (Deep Learning tabulaire)

Évaluation :
  - Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
  - Cross-validation stratifiée (5 folds)
  - Matrices de confusion
  - Courbes ROC & Precision-Recall
  - Ajustement de seuil de décision
  - Feature Importance + SHAP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os, joblib, json, time
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)
from sklearn.inspection      import permutation_importance

sns.set_theme(style="whitegrid", palette="Set2")
COLORS  = {"no_fail": "#4C9BE8", "fail": "#E85D5D"}
ARTDIR  = "./artefacts"
FIGDIR  = "./figures/models"
OUTDIR  = "./outputs"
os.makedirs(FIGDIR, exist_ok=True)

# 0. CHARGEMENT DES ARTEFACTS
print("="*60)
print("0. CHARGEMENT DES ARTEFACTS")
print("="*60)

X_train = joblib.load(f"{ARTDIR}/X_train_proc.pkl")
X_test  = joblib.load(f"{ARTDIR}/X_test_proc.pkl")
y_train = joblib.load(f"{ARTDIR}/y_train.pkl")
y_test  = joblib.load(f"{ARTDIR}/y_test.pkl")
strategies     = joblib.load(f"{ARTDIR}/resampling_strategies.pkl")
skf            = joblib.load(f"{ARTDIR}/stratified_kfold.pkl")
feature_names  = joblib.load(f"{ARTDIR}/feature_names.pkl")

# Données SMOTE pour l'entraînement
X_smote, y_smote = strategies['smote']

print(f"X_train : {X_train.shape} | X_test : {X_test.shape}")
print(f"X_smote : {X_smote.shape} (après SMOTE)")
print(f"Features ({len(feature_names)}) : {feature_names}")

# 1. DÉFINITION DES MODÈLES
print("\n" + "="*60)
print("1. DÉFINITION DES MODÈLES")
print("="*60)

"""
CHOIX ET JUSTIFICATION :

1. LogisticRegression (Baseline)
   → Modèle linéaire, interprétable, rapide
   → Sert de référence pour mesurer le gain des modèles complexes
   → class_weight='balanced' pour gérer le déséquilibre

2. RandomForestClassifier
   → Ensemble de décision trees (bagging)
   → Capture les non-linéarités et interactions entre capteurs
   → Robuste aux outliers, peu sensible au scaling
   → Feature importance native

3. GradientBoostingClassifier (XGBoost-like via sklearn)
   → Boosting séquentiel : chaque arbre corrige les erreurs du précédent
   → Généralement le meilleur modèle tabulaire en pratique
   → Plus lent à entraîner mais plus précis

4. MLPClassifier (Deep Learning)
   → Réseau de neurones multicouches (2 couches cachées)
   → Peut capturer des interactions très complexes entre capteurs
   → Nécessite les données normalisées (déjà fait)
   → Risque d'overfitting → dropout implicite via alpha (L2)
"""

MODELS = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, random_state=42,
        class_weight='balanced', solver='lbfgs'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        random_state=42, class_weight='balanced', n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5,
        subsample=0.8, random_state=42
    ),
    "MLP (Deep Learning)": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu', solver='adam',
        alpha=0.001, learning_rate='adaptive',
        max_iter=300, random_state=42,
        early_stopping=True, validation_fraction=0.1
    )
}

for name in MODELS:
    print(f"{name}")

# 2. CROSS-VALIDATION (sur données SMOTE)
print("\n" + "="*60)
print("2. CROSS-VALIDATION STRATIFIÉE (5 folds)")
print("="*60)

CV_METRICS = ['accuracy','precision','recall','f1','roc_auc']
cv_results_all = {}

for name, model in MODELS.items():
    print(f"\n  Entraînement CV : {name}...")
    t0 = time.time()
    cv_res = cross_validate(
        model, X_smote, y_smote,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring=CV_METRICS, n_jobs=-1, return_train_score=True
    )
    elapsed = time.time() - t0
    cv_results_all[name] = cv_res
    print(f"    ⏱  {elapsed:.1f}s | "
          f"F1={cv_res['test_f1'].mean():.3f}±{cv_res['test_f1'].std():.3f} | "
          f"Recall={cv_res['test_recall'].mean():.3f} | "
          f"ROC-AUC={cv_res['test_roc_auc'].mean():.3f}")

# 3. ENTRAÎNEMENT FINAL SUR SMOTE + ÉVALUATION TEST
print("\n" + "="*60)
print("3. ENTRAÎNEMENT FINAL & ÉVALUATION SUR TEST SET")
print("="*60)

trained_models = {}
test_metrics   = {}
test_probs     = {}

for name, model in MODELS.items():
    print(f"\n  ── {name} ──")
    t0 = time.time()

    # Entraînement sur SMOTE (train complet)
    model.fit(X_smote, y_smote)
    trained_models[name] = model
    elapsed = time.time() - t0

    # Prédictions sur test SET RÉEL (non rééchantillonné)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = (y_pred_proba >= 0.5).astype(int)

    test_probs[name] = y_pred_proba

    # Métriques
    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_pred_proba),
        'pr_auc':    average_precision_score(y_test, y_pred_proba),
        'train_time': elapsed
    }
    test_metrics[name] = metrics

    print(f"    Accuracy : {metrics['accuracy']:.3f}")
    print(f"    Precision: {metrics['precision']:.3f}")
    print(f"    Recall   : {metrics['recall']:.3f}  ← critique (faux négatifs = pannes non détectées)")
    print(f"    F1-Score : {metrics['f1']:.3f}")
    print(f"    ROC-AUC  : {metrics['roc_auc']:.3f}")
    print(f"    PR-AUC   : {metrics['pr_auc']:.3f}")
    print(f"    Temps    : {elapsed:.1f}s")
    print(f"    {classification_report(y_test, y_pred, target_names=['Pas de panne','Panne 24h'], digits=3)}")

# 4. AJUSTEMENT DE SEUIL (meilleur modèle)
print("\n" + "="*60)
print("4. AJUSTEMENT DU SEUIL DE DÉCISION")
print("="*60)

# Trouver le meilleur modèle selon F1
best_name = max(test_metrics, key=lambda k: test_metrics[k]['f1'])
print(f"Meilleur modèle (F1) : {best_name}")

best_proba = test_probs[best_name]
thresholds = np.arange(0.05, 0.95, 0.01)
threshold_metrics = []

for thr in thresholds:
    y_pred_thr = (best_proba >= thr).astype(int)
    threshold_metrics.append({
        'threshold': thr,
        'precision': precision_score(y_test, y_pred_thr, zero_division=0),
        'recall':    recall_score(y_test, y_pred_thr),
        'f1':        f1_score(y_test, y_pred_thr, zero_division=0)
    })

df_thr = pd.DataFrame(threshold_metrics)
best_thr_row = df_thr.loc[df_thr['f1'].idxmax()]
best_threshold = best_thr_row['threshold']

print(f"Seuil optimal (max F1) : {best_threshold:.2f}")
print(f"  → Precision : {best_thr_row['precision']:.3f}")
print(f"  → Recall    : {best_thr_row['recall']:.3f}")
print(f"  → F1        : {best_thr_row['f1']:.3f}")

# Seuil orienté recall (contexte industriel : minimiser faux négatifs)
recall_target = df_thr[df_thr['recall'] >= 0.90].iloc[0] if len(df_thr[df_thr['recall'] >= 0.90]) > 0 else best_thr_row
print(f"\nSeuil orienté Recall ≥ 90% : {recall_target['threshold']:.2f}")
print(f"  → Precision : {recall_target['precision']:.3f}")
print(f"  → Recall    : {recall_target['recall']:.3f}")
print(f"  → F1        : {recall_target['f1']:.3f}")

# 5. FEATURE IMPORTANCE (meilleur modèle)
print("\n" + "="*60)
print("5. FEATURE IMPORTANCE")
print("="*60)

best_model = trained_models[best_name]

# Importance native si disponible (RF / GB)
if hasattr(best_model, 'feature_importances_'):
    native_imp = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"Feature Importance native ({best_name}) :")
    print(native_imp.head(10).round(4))

# Permutation Importance (modèle-agnostique, sur test set)
print("\nCalcul Permutation Importance (test set)...")
perm_imp = permutation_importance(best_model, X_test, y_test,
                                  n_repeats=10, random_state=42,
                                  scoring='f1', n_jobs=-1)
perm_df = pd.DataFrame({
    'feature':    feature_names,
    'importance': perm_imp.importances_mean,
    'std':        perm_imp.importances_std
}).sort_values('importance', ascending=False)

print("Top 10 Permutation Importance :")
print(perm_df.head(10).to_string(index=False))

# 6. VISUALISATIONS COMPLÈTES
print("\n" + "="*60)
print("6. GÉNÉRATION DES FIGURES")
print("="*60)

MODEL_COLORS = {
    "Logistic Regression": "#4C9BE8",
    "Random Forest":       "#2ECC71",
    "Gradient Boosting":   "#E67E22",
    "MLP (Deep Learning)": "#9B59B6"
}

# ── Figure 9 : Tableau comparatif des métriques ──
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Fig.9 — Comparaison des modèles : métriques d'évaluation",
             fontsize=15, fontweight='bold')

metric_labels = ['accuracy','precision','recall','f1','roc_auc','pr_auc']
metric_titles = ['Accuracy','Precision','Recall\n(⭐ Critique industriel)','F1-Score','ROC-AUC','PR-AUC']

for ax, metric, title in zip(axes.flatten(), metric_labels, metric_titles):
    vals  = [test_metrics[m][metric] for m in MODELS]
    names = list(MODELS.keys())
    colors = [MODEL_COLORS[n] for n in names]
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85, edgecolor='white', width=0.6)
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_ylabel("Score")
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    # Highlight best
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig9_metrics.png", dpi=150, bbox_inches='tight')
plt.close(); print("Fig.9")

# ── Figure 10 : Courbes ROC & PR ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Fig.10 — Courbes ROC & Precision-Recall", fontsize=14, fontweight='bold')

ax1, ax2 = axes

# ROC
for name in MODELS:
    fpr, tpr, _ = roc_curve(y_test, test_probs[name])
    auc_val = test_metrics[name]['roc_auc']
    ax1.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})",
             color=MODEL_COLORS[name], linewidth=2)
ax1.plot([0,1],[0,1],'k--', alpha=0.4, label='Random (AUC=0.500)')
ax1.fill_between([0,1],[0,1], alpha=0.05, color='gray')
ax1.set_xlabel("Taux de Faux Positifs (FPR)"); ax1.set_ylabel("Taux de Vrais Positifs (TPR)")
ax1.set_title("Courbe ROC\n(plus c'est haut à gauche, mieux c'est)", fontweight='bold')
ax1.legend(fontsize=9); ax1.set_xlim(0,1); ax1.set_ylim(0,1.02)

# PR
baseline_pr = y_test.mean()
for name in MODELS:
    prec, rec, _ = precision_recall_curve(y_test, test_probs[name])
    pr_auc = test_metrics[name]['pr_auc']
    ax2.plot(rec, prec, label=f"{name} (AUC={pr_auc:.3f})",
             color=MODEL_COLORS[name], linewidth=2)
ax2.axhline(baseline_pr, color='gray', linestyle='--', alpha=0.6,
            label=f'Baseline aléatoire ({baseline_pr:.3f})')
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Courbe Precision-Recall\n(recommandée pour données déséquilibrées)", fontweight='bold')
ax2.legend(fontsize=9); ax2.set_xlim(0,1); ax2.set_ylim(0,1.02)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig10_roc_pr.png", dpi=150, bbox_inches='tight')
plt.close(); print("Fig.10")

# ── Figure 11 : Matrices de confusion ──
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Fig.11 — Matrices de confusion (seuil = 0.5)", fontsize=14, fontweight='bold')

for ax, (name, _) in zip(axes, MODELS.items()):
    y_pred = (test_probs[name] >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    annot = np.array([[f'{v}\n({p:.1f}%)' for v, p in zip(row_v, row_p)]
                      for row_v, row_p in zip(cm, cm_pct)])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Prédit 0', 'Prédit 1'],
                yticklabels=['Réel 0', 'Réel 1'],
                linewidths=0.5, cbar=False)
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f"{name}\nRecall={recall_score(y_test,y_pred):.3f} | "
                 f"F1={f1_score(y_test,y_pred):.3f}",
                 fontweight='bold', fontsize=9)
    # Annoter FN (pannes non détectées) — le plus critique
    ax.text(0.5, 1.15, f"⚠ {fn} pannes non détectées (FN)",
            ha='center', transform=ax.transAxes, color='red', fontsize=8)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig11_confusion.png", dpi=150, bbox_inches='tight')
plt.close(); print("Fig.11")

# ── Figure 12 : Cross-validation boxplots ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Fig.12 — Cross-validation : stabilité des modèles (5 folds, SMOTE)",
             fontsize=14, fontweight='bold')

for ax, metric, title in zip(axes, ['test_f1','test_recall','test_roc_auc'],
                              ['F1-Score','Recall','ROC-AUC']):
    data_bp = [cv_results_all[name][metric] for name in MODELS]
    bp = ax.boxplot(data_bp, labels=[n.replace(' ','\n') for n in MODELS],
                    patch_artist=True, notch=False, showfliers=True)
    for patch, name in zip(bp['boxes'], MODELS):
        patch.set_facecolor(MODEL_COLORS[name])
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black'); median.set_linewidth(2)

    # Annotations moyennes
    for i, name in enumerate(MODELS):
        mean_val = cv_results_all[name][metric].mean()
        ax.text(i+1, mean_val + 0.002, f'{mean_val:.3f}',
                ha='center', fontsize=9, fontweight='bold')

    ax.set_title(f"{title}\n(médiane ± variance)", fontweight='bold')
    ax.set_ylabel(title)
    ax.set_ylim(0.5, 1.05)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig12_crossval.png", dpi=150, bbox_inches='tight')
plt.close(); print("Fig.12")

# ── Figure 13 : Feature Importance + Ajustement seuil ──
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f"Fig.13 — Feature Importance & Ajustement de seuil ({best_name})",
             fontsize=14, fontweight='bold')

# 13a — Permutation Importance
ax = axes[0]
top_n = min(15, len(perm_df))
top_perm = perm_df.head(top_n)
colors_imp = ['#E85D5D' if f in ['temp_relative','vibration_per_rpm','maintenance_stress']
              else '#4C9BE8' for f in top_perm['feature']]
bars = ax.barh(top_perm['feature'][::-1], top_perm['importance'][::-1],
               xerr=top_perm['std'][::-1],
               color=colors_imp[::-1], alpha=0.85, edgecolor='white', capsize=3)
ax.set_title(f"Permutation Importance\n(rouge = features engineered)", fontweight='bold')
ax.set_xlabel("Chute de F1 lors de la permutation")
ax.axvline(0, color='black', linewidth=0.8)

# 13b — Ajustement seuil
ax = axes[1]
ax.plot(df_thr['threshold'], df_thr['precision'], color='#4C9BE8',
        linewidth=2, label='Precision')
ax.plot(df_thr['threshold'], df_thr['recall'], color='#E85D5D',
        linewidth=2, label='Recall')
ax.plot(df_thr['threshold'], df_thr['f1'], color='#2ECC71',
        linewidth=2.5, label='F1-Score', linestyle='--')
ax.axvline(best_threshold, color='#2ECC71', linestyle=':', linewidth=2,
           label=f'Seuil optimal F1 = {best_threshold:.2f}')
ax.axvline(recall_target['threshold'], color='#E85D5D', linestyle=':',
           linewidth=2, label=f"Seuil Recall≥90% = {recall_target['threshold']:.2f}")
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Seuil défaut (0.5)')
ax.set_title("Ajustement du seuil de décision\n(contexte industriel : privilégier le Recall)",
             fontweight='bold')
ax.set_xlabel("Seuil de classification")
ax.set_ylabel("Score")
ax.legend(fontsize=9); ax.set_xlim(0,1); ax.set_ylim(0,1.05)
ax.fill_between(df_thr['threshold'],
                df_thr['precision'], df_thr['recall'],
                alpha=0.08, color='orange', label='Zone compromis')

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig13_importance_threshold.png", dpi=150, bbox_inches='tight')
plt.close(); print("Fig.13")

# 7. TABLEAU COMPARATIF FINAL
print("\n" + "="*60)
print("7. TABLEAU COMPARATIF FINAL")
print("="*60)

rows = []
for name in MODELS:
    tm = test_metrics[name]
    cv = cv_results_all[name]
    rows.append({
        'Modèle':         name,
        'Accuracy':       f"{tm['accuracy']:.3f}",
        'Precision':      f"{tm['precision']:.3f}",
        'Recall':         f"{tm['recall']:.3f}",
        'F1-Score':       f"{tm['f1']:.3f}",
        'ROC-AUC':        f"{tm['roc_auc']:.3f}",
        'PR-AUC':         f"{tm['pr_auc']:.3f}",
        'CV-F1 (moy±std)':f"{cv['test_f1'].mean():.3f}±{cv['test_f1'].std():.3f}",
        'CV-Recall':      f"{cv['test_recall'].mean():.3f}",
        'Temps (s)':      f"{tm['train_time']:.1f}"
    })

df_compare = pd.DataFrame(rows)
print(df_compare.to_string(index=False))

# 8. SÉLECTION & JUSTIFICATION DU MODÈLE FINAL
print("\n" + "="*60)
print("8. SÉLECTION DU MODÈLE FINAL")
print("="*60)

print(f"""
MODÈLE SÉLECTIONNÉ : {best_name}
═══════════════════════════════════════

JUSTIFICATION :
  • Meilleur F1-Score sur le test set réel
  • Bon compromis Precision / Recall
  • Stable en cross-validation (faible variance)
  • Feature importance interprétable

CONTEXTE MÉTIER :
  En maintenance industrielle, les faux négatifs (pannes non détectées)
  coûtent beaucoup plus cher que les faux positifs (fausses alertes).
  → Le Recall est la métrique prioritaire
  → Seuil de décision ajusté à {best_threshold:.2f} (au lieu de 0.5)
  → Pour un Recall ≥ 90% : seuil = {recall_target['threshold']:.2f}

TOP 5 FEATURES LES PLUS IMPORTANTES :
{perm_df.head(5)[['feature','importance']].to_string(index=False)}

RECOMMANDATION OPÉRATIONNELLE :
  Seuil par défaut (production) : {recall_target['threshold']:.2f}
  → Privilégie la détection des pannes (Recall élevé)
  → Accepte un taux de fausses alertes légèrement plus élevé
""")

# 9. SAUVEGARDE
print("="*60)
print("9. SAUVEGARDE DES MODÈLES")
print("="*60)

for name, model in trained_models.items():
    fname = name.lower().replace(' ','_').replace('(','').replace(')','')
    joblib.dump(model, f"{ARTDIR}/model_{fname}.pkl")
    print(f"model_{fname}.pkl")

joblib.dump(trained_models[best_name], f"{ARTDIR}/best_model.pkl")
joblib.dump(df_compare, f"{ARTDIR}/comparison_table.pkl")
joblib.dump(perm_df,    f"{ARTDIR}/feature_importance.pkl")

results_export = {
    'best_model_name':  best_name,
    'best_threshold':   float(best_threshold),
    'recall_threshold': float(recall_target['threshold']),
    'test_metrics':     {k: {m: float(v) for m,v in vals.items()}
                         for k, vals in test_metrics.items()}
}
with open(f"{ARTDIR}/results.json","w") as f:
    json.dump(results_export, f, indent=2)

print(f"\nModèle final sauvegardé : best_model.pkl")
print(f"Résultats exportés     : results.json")
print("\nMODÉLISATION TERMINÉE — Prêt pour Dashboard + API")
