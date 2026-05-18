"""
=============================================================
EDA COMPLÈTE — Maintenance Prédictive Industrielle
Tâche : Prédiction de Panne dans les 24h (Classification Binaire)
Variable cible : failure_within_24h
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style global ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
COLORS = {"no_fail": "#4C9BE8", "fail": "#E85D5D"}
TARGET = "failure_within_24h"
OUTPUT = "./figures/eda"
import os; os.makedirs(OUTPUT, exist_ok=True)

# ── Chargement des données ────────────────────────────────────
df = pd.read_csv('./industrial_machine_maintenance.csv',
                 parse_dates=['timestamp'])

NUMERIC_SENSORS = ['vibration_rms', 'temperature_motor', 'current_phase_avg',
                   'pressure_level', 'rpm', 'hours_since_maintenance',
                   'ambient_temp', 'rul_hours']

print("✅ Dataset chargé :", df.shape)

# =============================================================
# FIGURE 1 — Vue d'ensemble du dataset
# =============================================================
fig = plt.figure(figsize=(18, 10))
fig.suptitle("Fig.1 — Vue d'ensemble du dataset", fontsize=16, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
counts = df[TARGET].value_counts()
bars = ax1.bar(['Pas de panne\n(0)', 'Panne < 24h\n(1)'],
               counts.values,
               color=[COLORS['no_fail'], COLORS['fail']], edgecolor='white', width=0.5)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
             f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax1.set_title("Répartition de la variable cible", fontweight='bold')
ax1.set_ylabel("Nombre d'observations")
ax1.set_ylim(0, 24000)

ax2 = fig.add_subplot(gs[0, 1])
missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
missing = missing[missing > 0]
colors_m = ['#FF9999' if v > 4 else '#FFD580' for v in missing.values]
bars2 = ax2.barh(missing.index, missing.values, color=colors_m, edgecolor='white')
for bar, val in zip(bars2, missing.values):
    ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9)
ax2.set_title("Valeurs manquantes (%)", fontweight='bold')
ax2.set_xlabel("% de valeurs manquantes")
ax2.axvline(5, color='red', linestyle='--', alpha=0.5, label='Seuil 5%')
ax2.legend(fontsize=8)

ax3 = fig.add_subplot(gs[0, 2])
failure_counts = df['failure_type'].value_counts()
palette = ['#cccccc'] + list(sns.color_palette("Set2", len(failure_counts)-1))
ax3.pie(failure_counts.values, labels=failure_counts.index,
        colors=palette, autopct='%1.1f%%', startangle=140,
        textprops={'fontsize': 9})
ax3.set_title("Distribution des types de pannes", fontweight='bold')

ax4 = fig.add_subplot(gs[1, 0])
fail_by_type = df.groupby('machine_type')[TARGET].mean().sort_values() * 100
bars4 = ax4.bar(fail_by_type.index, fail_by_type.values,
                color=sns.color_palette("Set2", len(fail_by_type)), edgecolor='white')
for bar, val in zip(bars4, fail_by_type.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax4.set_title("Taux de panne par type de machine", fontweight='bold')
ax4.set_ylabel("Taux de panne (%)")
ax4.set_ylim(0, 22)

ax5 = fig.add_subplot(gs[1, 1])
fail_by_mode = df.groupby('operating_mode')[TARGET].mean().sort_values() * 100
bars5 = ax5.bar(fail_by_mode.index, fail_by_mode.values,
                color=['#4C9BE8', '#F4A261', '#E85D5D'], edgecolor='white')
for bar, val in zip(bars5, fail_by_mode.values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax5.set_title("Taux de panne par mode opératoire", fontweight='bold')
ax5.set_ylabel("Taux de panne (%)")
ax5.set_ylim(0, 22)

ax6 = fig.add_subplot(gs[1, 2])
machine_counts = df.groupby('machine_id')[TARGET].agg(['count', 'sum'])
machine_counts['fail_rate'] = machine_counts['sum'] / machine_counts['count'] * 100
ax6.bar(machine_counts.index, machine_counts['count'], color='#94C5F8', label='Total', alpha=0.7)
ax6_twin = ax6.twinx()
ax6_twin.plot(machine_counts.index, machine_counts['fail_rate'], 'r.-', label='Taux panne', linewidth=1.5)
ax6.set_title("Volume & taux de panne par machine", fontweight='bold')
ax6.set_xlabel("Machine ID")
ax6.set_ylabel("Nb d'observations", color='#4C9BE8')
ax6_twin.set_ylabel("Taux de panne (%)", color='red')

plt.savefig(f"{OUTPUT}/fig1_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("Fig.1 générée")

# =============================================================
# FIGURE 2 — Distributions des capteurs (classe 0 vs 1)
# =============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Fig.2 — Distributions des capteurs par classe\n(Bleu = Pas de panne | Rouge = Panne dans 24h)",
             fontsize=14, fontweight='bold')

for ax, col in zip(axes.flatten(), NUMERIC_SENSORS):
    df0 = df[df[TARGET]==0][col].dropna()
    df1 = df[df[TARGET]==1][col].dropna()

    ax.hist(df0, bins=50, alpha=0.6, color=COLORS['no_fail'], density=True, label='Pas de panne')
    ax.hist(df1, bins=50, alpha=0.6, color=COLORS['fail'], density=True, label='Panne 24h')

    ax.axvline(df0.mean(), color=COLORS['no_fail'], linestyle='--', linewidth=1.5,
               label=f'Moy. 0: {df0.mean():.1f}')
    ax.axvline(df1.mean(), color=COLORS['fail'], linestyle='--', linewidth=1.5,
               label=f'Moy. 1: {df1.mean():.1f}')

    ax.set_title(col, fontweight='bold', fontsize=10)
    ax.set_xlabel(col)
    ax.legend(fontsize=7)
    ax.set_ylabel("Densité")

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig2_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Fig.2 générée")

# =============================================================
# FIGURE 3 — Boxplots capteurs vs cible
# =============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle("Fig.3 — Boxplots : signaux capteurs vs variable cible",
             fontsize=14, fontweight='bold')

for ax, col in zip(axes.flatten(), NUMERIC_SENSORS):
    data_plot = [df[df[TARGET]==0][col].dropna(), df[df[TARGET]==1][col].dropna()]
    bp = ax.boxplot(data_plot, labels=['Pas de panne', 'Panne 24h'],
                    patch_artist=True, notch=True, showfliers=False)
    bp['boxes'][0].set_facecolor(COLORS['no_fail'])
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(COLORS['fail'])
    bp['boxes'][1].set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    ax.set_title(col, fontweight='bold', fontsize=10)
    ax.set_ylabel(col)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig3_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()
print("Fig.3 générée")

# =============================================================
# FIGURE 4 — Matrice de corrélation
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Fig.4 — Analyse des corrélations", fontsize=14, fontweight='bold')

corr_cols = NUMERIC_SENSORS + [TARGET]
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=axes[0], linewidths=0.5,
            annot_kws={'fontsize': 8})
axes[0].set_title("Matrice de corrélation complète", fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

corr_target = df[corr_cols].corr()[TARGET].drop(TARGET).sort_values()
colors_corr = [COLORS['fail'] if v > 0 else COLORS['no_fail'] for v in corr_target.values]
axes[1].barh(corr_target.index, corr_target.values, color=colors_corr, edgecolor='white')
axes[1].axvline(0, color='black', linewidth=0.8)
for i, (idx, val) in enumerate(corr_target.items()):
    axes[1].text(val + (0.005 if val >= 0 else -0.005), i,
                 f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
axes[1].set_title(f"Corrélations avec {TARGET}", fontweight='bold')
axes[1].set_xlabel("Coefficient de corrélation de Pearson")

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig4_correlations.png", dpi=150, bbox_inches='tight')
plt.close()
print("Fig.4 générée")

# =============================================================
# FIGURE 5 — Analyse du déséquilibre & stratégie
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Fig.5 — Déséquilibre des classes & analyse", fontsize=14, fontweight='bold')

ax = axes[0]
ratio = df[TARGET].value_counts()
wedges, texts, autotexts = ax.pie(ratio.values,
       labels=['Pas de panne (85.2%)', 'Panne 24h (14.8%)'],
       colors=[COLORS['no_fail'], COLORS['fail']],
       autopct='%1.1f%%', startangle=90,
       explode=(0, 0.05), shadow=True)
ax.set_title(f"Déséquilibre : ratio {ratio[0]//ratio[1]}:1\n→ Nécessite une stratégie adaptée",
             fontweight='bold')

ax = axes[1]
pivot = df.pivot_table(values=TARGET, index='machine_type',
                       columns='operating_mode', aggfunc='mean') * 100
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Taux de panne (%)'})
ax.set_title("Taux de panne (%) :\ntype machine × mode opératoire", fontweight='bold')

ax = axes[2]
df['month'] = df['timestamp'].dt.to_period('M')
monthly = df.groupby('month')[TARGET].mean() * 100
monthly.index = monthly.index.astype(str)
ax.plot(range(len(monthly)), monthly.values, 'o-', color=COLORS['fail'],
        linewidth=2, markersize=5)
ax.fill_between(range(len(monthly)), monthly.values, alpha=0.15, color=COLORS['fail'])
ax.axhline(monthly.mean(), color='gray', linestyle='--', label=f'Moy. globale: {monthly.mean():.1f}%')
ax.set_title("Évolution mensuelle du taux de panne", fontweight='bold')
ax.set_xlabel("Mois")
ax.set_ylabel("Taux de panne (%)")
ax.set_xticks(range(0, len(monthly), 2))
ax.set_xticklabels([monthly.index[i] for i in range(0, len(monthly), 2)], rotation=45, fontsize=8)
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig5_imbalance.png", dpi=150, bbox_inches='tight')
plt.close()
print("Fig.5 générée")

# =============================================================
# FIGURE 6 — Scatter plots & relations entre capteurs clés
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Fig.6 — Relations entre capteurs clés (coloré par cible)",
             fontsize=14, fontweight='bold')

pairs = [
    ('vibration_rms', 'temperature_motor'),
    ('rul_hours', 'temperature_motor'),
    ('vibration_rms', 'rul_hours')
]

for ax, (x_col, y_col) in zip(axes, pairs):
    df_sample = df.dropna(subset=[x_col, y_col]).sample(min(3000, len(df)), random_state=42)
    df0 = df_sample[df_sample[TARGET]==0]
    df1 = df_sample[df_sample[TARGET]==1]
    ax.scatter(df0[x_col], df0[y_col], alpha=0.3, s=10,
               color=COLORS['no_fail'], label='Pas de panne')
    ax.scatter(df1[x_col], df1[y_col], alpha=0.5, s=15,
               color=COLORS['fail'], label='Panne 24h')
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_title(f"{x_col}\nvs {y_col}", fontweight='bold')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig6_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("Fig.6 générée")

# =============================================================
# FIGURE 7 — Analyse des outliers
# =============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle("Fig.7 — Détection et analyse des outliers (méthode IQR)",
             fontsize=14, fontweight='bold')

for ax, col in zip(axes.flatten(), NUMERIC_SENSORS):
    data = df[col].dropna()
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers_mask = (data < low) | (data > high)
    n_out = outliers_mask.sum()

    ax.hist(data[~outliers_mask], bins=40, alpha=0.7, color='#94C5F8', label='Normal')
    if n_out > 0:
        ax.hist(data[outliers_mask], bins=20, alpha=0.8, color='#FF9999', label=f'Outliers ({n_out})')
    ax.axvline(low, color='orange', linestyle='--', linewidth=1, label=f'Borne basse')
    ax.axvline(high, color='red', linestyle='--', linewidth=1, label=f'Borne haute')
    ax.set_title(f"{col}\n{n_out} outliers ({n_out/len(df)*100:.1f}%)", fontweight='bold', fontsize=9)
    ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig7_outliers.png", dpi=150, bbox_inches='tight')
plt.close()
print("Fig.7 générée")

# =============================================================
# SYNTHÈSE STATISTIQUE
# =============================================================
print("\n" + "="*60)
print("SYNTHÈSE EDA — Points clés")
print("="*60)
print(f"""
DATASET
  • {df.shape[0]:,} observations | {df.shape[1]} variables
  • Période : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}
  • 20 machines | 4 types (CNC, Pump, Compressor, Robotic Arm)

VARIABLE CIBLE : failure_within_24h
  • Classe 0 (pas de panne) : 20 482 ({20482/24042*100:.1f}%)
  • Classe 1 (panne < 24h)  : 3 560  ({3560/24042*100:.1f}%)
  • Ratio déséquilibre : ~5.75:1 → TRAITEMENT REQUIS

VALEURS MANQUANTES
  • vibration_rms    : 4.2%
  • pressure_level   : 3.8%
  • temperature_motor: 3.5%
  • current_phase_avg: 3.0%
  • rpm              : 2.2%
  → Imputation recommandée : médiane par machine_type

TOP CORRÉLATIONS AVEC LA CIBLE
  • temperature_motor  : +0.386 (signal fort)
  • vibration_rms      : +0.264 (signal fort)
  • rul_hours          : -0.253 (inverse : moins de vie = plus de risque)
  • current_phase_avg  : +0.157
  → ambient_temp, rpm, pressure_level : corrélation faible

 OUTLIERS
  • vibration_rms : 478 outliers (2.0%)
  • rpm           : 373 outliers (1.6%)
  → Impact à analyser : peuvent correspondre à de vraies pannes

RECOMMANDATIONS POUR LA SUITE
  1. Imputer les NaN (médiane par machine_type)
  2. Gérer le déséquilibre (class_weight + SMOTE)
  3. Encoder : machine_type, operating_mode (OneHot)
  4. Supprimer : timestamp, machine_id, failure_type (data leakage potentiel)
  5. Conserver rul_hours mais surveiller (proxy de la cible)
  6. Normaliser pour LR et MLP, pas nécessaire pour RF/XGB
""")

print("Toutes les figures générées dans:", OUTPUT)
