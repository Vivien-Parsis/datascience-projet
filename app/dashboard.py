"""
=============================================================
DASHBOARD — Maintenance Prédictive Industrielle
Streamlit | Orienté utilisateur métier (responsable maintenance)
=============================================================
Lancement : streamlit run dashboard.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, json, joblib
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "🔧 Maintenance Prédictive",
    page_icon  = "🔧",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

API_URL  = "http://localhost:8000"
ARTDIR   = "../artefacts"

# ─────────────────────────────────────────────────────────────
# STYLES CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #1e3a5f, #2980b9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #666; font-size: 1rem; margin-bottom: 2rem; }
    .kpi-card {
        background: white; border-radius: 12px;
        padding: 1.2rem 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #2980b9; margin-bottom: 1rem;
    }
    .kpi-value { font-size: 2rem; font-weight: 800; color: #1e3a5f; }
    .kpi-label { font-size: 0.85rem; color: #888; margin-top: 0.2rem; }
    .risk-critical { background: #fff0f0; border-left-color: #e74c3c; }
    .risk-high     { background: #fff7f0; border-left-color: #e67e22; }
    .risk-moderate { background: #fffbf0; border-left-color: #f39c12; }
    .risk-low      { background: #f0fff4; border-left-color: #27ae60; }
    .alert-box {
        padding: 1rem 1.5rem; border-radius: 10px;
        font-size: 1.1rem; font-weight: 600; margin: 1rem 0;
    }
    .alert-danger  { background: #ffe0e0; color: #c0392b; border: 1px solid #e74c3c; }
    .alert-success { background: #e0ffe8; color: #1e8449; border: 1px solid #27ae60; }
    .section-title {
        font-size: 1.2rem; font-weight: 700; color: #1e3a5f;
        border-bottom: 2px solid #2980b9; padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES (sans API pour les graphes globaux)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('../industrial_machine_maintenance.csv',
                     parse_dates=['timestamp'])
    return df

@st.cache_resource
def load_artefacts():
    try:
        comp  = joblib.load(f"{ARTDIR}/comparison_table.pkl")
        fimp  = joblib.load(f"{ARTDIR}/feature_importance.pkl")
        proba_test = joblib.load(f"{ARTDIR}/best_model.pkl")
        y_test     = joblib.load(f"{ARTDIR}/y_test.pkl")
        X_test     = joblib.load(f"{ARTDIR}/X_test_proc.pkl")
        preprocessor = joblib.load(f"{ARTDIR}/preprocessor.pkl")
        best_model   = joblib.load(f"{ARTDIR}/best_model.pkl")
        with open(f"{ARTDIR}/results.json") as f:
            results = json.load(f)
        return comp, fimp, y_test, X_test, preprocessor, best_model, results
    except Exception as e:
        return None, None, None, None, None, None, {}

df_raw = load_data()
comp_df, feat_imp, y_test, X_test, preprocessor, best_model, results = load_artefacts()

# ─────────────────────────────────────────────────────────────
# SIDEBAR — Navigation
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/maintenance.png", width=70)
    st.markdown("### 🔧 Maintenance Prédictive")
    st.caption("Système intelligent de détection de pannes")
    st.divider()

    page = st.radio("Navigation", [
        "🏠 Vue d'ensemble",
        "🔮 Simulateur de panne",
        "📊 Comparaison des modèles",
        "🔍 Importance des variables",
        "⚙️ API & Infos système"
    ])

    st.divider()
    # Statut API
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("🟢 API opérationnelle")
        else:
            st.warning("🟡 API — réponse inattendue")
    except:
        st.error("🔴 API hors ligne\nLancez : `uvicorn api:app`")

    st.caption(f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")

# ─────────────────────────────────────────────────────────────
# PAGE 1 — VUE D'ENSEMBLE
# ─────────────────────────────────────────────────────────────
if page == "🏠 Vue d'ensemble":
    st.markdown('<div class="main-title">🔧 Dashboard Maintenance Prédictive</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Système intelligent de surveillance et prédiction de pannes industrielles</div>', unsafe_allow_html=True)

    # KPIs principaux
    total    = len(df_raw)
    n_fail   = df_raw['failure_within_24h'].sum()
    n_safe   = total - n_fail
    fail_pct = n_fail / total * 100
    n_mach   = df_raw['machine_id'].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{total:,}</div>
            <div class="kpi-label">📋 Enregistrements totaux</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card risk-critical">
            <div class="kpi-value" style="color:#e74c3c">{n_fail:,}</div>
            <div class="kpi-label">⚠️ Pannes détectées ({fail_pct:.1f}%)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card risk-low">
            <div class="kpi-value" style="color:#27ae60">{n_safe:,}</div>
            <div class="kpi-label">✅ Machines saines</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{n_mach}</div>
            <div class="kpi-label">🏭 Machines surveillées</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">📈 Taux de panne par type de machine</div>', unsafe_allow_html=True)
        fail_by_type = df_raw.groupby('machine_type')['failure_within_24h'].agg(['mean','sum','count']).reset_index()
        fail_by_type['Taux (%)'] = (fail_by_type['mean'] * 100).round(1)
        fig = px.bar(fail_by_type, x='machine_type', y='Taux (%)',
                     color='Taux (%)', color_continuous_scale='RdYlGn_r',
                     text='Taux (%)', labels={'machine_type': 'Type de machine'},
                     title="Taux de panne (%) par type de machine")
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=320, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">🕐 Taux de panne par mode opératoire</div>', unsafe_allow_html=True)
        fail_by_mode = df_raw.groupby('operating_mode')['failure_within_24h'].mean().reset_index()
        fail_by_mode.columns = ['Mode', 'Taux']
        fail_by_mode['Taux (%)'] = (fail_by_mode['Taux'] * 100).round(1)
        fig2 = px.pie(fail_by_mode, values='Taux (%)', names='Mode',
                      color_discrete_sequence=['#2980b9','#27ae60','#e74c3c'],
                      title="Répartition du risque par mode")
        fig2.update_traces(textinfo='label+percent', pull=[0.05, 0, 0.1])
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # Distribution des capteurs clés
    st.markdown('<div class="section-title">📡 Distribution des capteurs clés par classe</div>', unsafe_allow_html=True)
    sensors = ['temperature_motor', 'vibration_rms', 'rul_hours', 'hours_since_maintenance']
    sensor_labels = ['Température moteur (°C)', 'Vibration RMS (mm/s)', 'Durée de vie restante (h)', 'Heures sans maintenance']

    fig3 = make_subplots(rows=1, cols=4, subplot_titles=sensor_labels)
    for i, (sensor, label) in enumerate(zip(sensors, sensor_labels), 1):
        df0 = df_raw[df_raw['failure_within_24h']==0][sensor].dropna()
        df1 = df_raw[df_raw['failure_within_24h']==1][sensor].dropna()
        fig3.add_trace(go.Histogram(x=df0, name='Pas de panne', opacity=0.65,
                       marker_color='#4C9BE8', showlegend=(i==1),
                       nbinsx=30), row=1, col=i)
        fig3.add_trace(go.Histogram(x=df1, name='Panne 24h', opacity=0.65,
                       marker_color='#E85D5D', showlegend=(i==1),
                       nbinsx=30), row=1, col=i)

    fig3.update_layout(height=300, barmode='overlay',
                       legend=dict(orientation='h', y=1.15))
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 2 — SIMULATEUR DE PANNE
# ─────────────────────────────────────────────────────────────
elif page == "🔮 Simulateur de panne":
    st.markdown('<div class="main-title">🔮 Simulateur de Panne</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Saisissez les paramètres de votre machine pour obtenir une prédiction en temps réel</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown('<div class="section-title">⚙️ Paramètres de la machine</div>', unsafe_allow_html=True)

        machine_type   = st.selectbox("🏭 Type de machine",   ["CNC", "Pump", "Compressor", "Robotic Arm"])
        operating_mode = st.selectbox("🔄 Mode opératoire",   ["idle", "normal", "peak"])

        st.markdown("**📡 Capteurs physiques**")
        c1, c2 = st.columns(2)
        with c1:
            vibration  = st.slider("Vibration RMS (mm/s)",        0.0, 12.0, 1.5, 0.1)
            temp_motor = st.slider("Température moteur (°C)",      20.0, 110.0, 55.0, 0.5)
            pressure   = st.slider("Pression (bar)",               15.0, 40.0, 23.0, 0.1)
        with c2:
            rpm        = st.slider("RPM",                          300.0, 2000.0, 900.0, 10.0)
            current    = st.slider("Courant moyen (A)",            3.0, 15.0, 6.0, 0.1)
            rul        = st.slider("Durée de vie restante (h)",    0.0, 200.0, 50.0, 1.0)

        st.markdown("**🌡️ Contexte**")
        c3, c4 = st.columns(2)
        with c3:
            ambient    = st.slider("Température ambiante (°C)",    10.0, 40.0, 20.0, 0.5)
        with c4:
            hours_maint= st.slider("Heures depuis maintenance",    0.0, 800.0, 250.0, 5.0)

        predict_btn = st.button("🚀 Lancer la prédiction", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-title">📊 Résultat de la prédiction</div>', unsafe_allow_html=True)

        if predict_btn:
            payload = {
                "vibration_rms": vibration, "temperature_motor": temp_motor,
                "current_phase_avg": current, "pressure_level": pressure,
                "rpm": rpm, "hours_since_maintenance": hours_maint,
                "ambient_temp": ambient, "rul_hours": rul,
                "machine_type": machine_type, "operating_mode": operating_mode
            }
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if resp.status_code == 200:
                    r = resp.json()
                    proba = r['probability_failure']
                    pred  = r['prediction']

                    # Jauge de risque
                    fig_gauge = go.Figure(go.Indicator(
                        mode  = "gauge+number+delta",
                        value = proba * 100,
                        title = {'text': "Probabilité de panne (%)", 'font': {'size': 16}},
                        delta = {'reference': 15, 'valueformat': '.1f'},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar':  {'color': "#e74c3c" if pred == 1 else "#27ae60"},
                            'steps': [
                                {'range': [0, 20],   'color': '#d5f5e3'},
                                {'range': [20, 50],  'color': '#fdebd0'},
                                {'range': [50, 75],  'color': '#fad7a0'},
                                {'range': [75, 100], 'color': '#fadbd8'},
                            ],
                            'threshold': {'line': {'color': 'red','width': 4}, 'value': 50}
                        }
                    ))
                    fig_gauge.update_layout(height=280)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # Alerte
                    if pred == 1:
                        st.markdown(f'<div class="alert-box alert-danger">🚨 {r["label"]}<br><small>{r["recommendation"]}</small></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="alert-box alert-success">✅ {r["label"]}<br><small>{r["recommendation"]}</small></div>', unsafe_allow_html=True)

                    # Niveau de risque
                    st.metric("Niveau de risque", r['risk_level'])
                    st.metric("Seuil utilisé", f"{r['threshold_used']:.2f}")

                    # Top facteurs de risque
                    st.markdown("**🔍 Facteurs de risque principaux**")
                    factors_df = pd.DataFrame(r['top_risk_factors'])
                    fig_bar = px.bar(factors_df, x='importance', y='feature',
                                     orientation='h', color='importance',
                                     color_continuous_scale='RdYlGn_r',
                                     title="Importance des variables (modèle)")
                    fig_bar.update_layout(height=250, showlegend=False,
                                          coloraxis_showscale=False,
                                          yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)

                else:
                    st.error(f"Erreur API ({resp.status_code}) : {resp.json().get('detail','')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ API non joignable. Lancez d'abord : `uvicorn api:app --port 8000`")
                # Fallback local sans API
                st.info("💡 Mode hors-ligne : prédiction directe (sans API)")
                if preprocessor and best_model:
                    import pandas as pd_loc, numpy as np_loc
                    row = {
                        'vibration_rms': vibration, 'temperature_motor': temp_motor,
                        'current_phase_avg': current, 'pressure_level': pressure,
                        'rpm': rpm, 'hours_since_maintenance': hours_maint,
                        'ambient_temp': ambient, 'rul_hours': rul,
                        'machine_type': machine_type, 'operating_mode': operating_mode,
                        'temp_relative': temp_motor - ambient,
                        'vibration_per_rpm': vibration / max(rpm, 1),
                        'maintenance_stress': np_loc.log1p(hours_maint)
                    }
                    all_feats = ['vibration_rms','temperature_motor','current_phase_avg',
                                 'pressure_level','rpm','hours_since_maintenance','ambient_temp',
                                 'rul_hours','temp_relative','vibration_per_rpm','maintenance_stress',
                                 'machine_type','operating_mode']
                    df_in  = pd_loc.DataFrame([row])[all_feats]
                    X_proc = preprocessor.transform(df_in)
                    proba  = float(best_model.predict_proba(X_proc)[0, 1])
                    pred   = int(proba >= 0.05)
                    st.metric("Probabilité de panne", f"{proba*100:.1f}%")
                    if pred == 1:
                        st.error("🚨 Panne probable dans les 24h")
                    else:
                        st.success("✅ Pas de panne détectée")
        else:
            st.info("👈 Configurez les paramètres de la machine et cliquez sur **Lancer la prédiction**")
            # Scénarios prédéfinis
            st.markdown("**📋 Scénarios de démonstration**")
            scenarios = {
                "🔴 Machine critique": {"vibration_rms": 8.5, "temp": 85.0, "rul": 5.0, "hours": 600.0},
                "🟡 Machine à surveiller": {"vibration_rms": 3.0, "temp": 65.0, "rul": 30.0, "hours": 300.0},
                "🟢 Machine saine": {"vibration_rms": 0.8, "temp": 42.0, "rul": 120.0, "hours": 50.0},
            }
            for label, vals in scenarios.items():
                st.markdown(f"**{label}** — Vibration: {vals['vibration_rms']} mm/s | Temp: {vals['temp']}°C | RUL: {vals['rul']}h")


# ─────────────────────────────────────────────────────────────
# PAGE 3 — COMPARAISON DES MODÈLES
# ─────────────────────────────────────────────────────────────
elif page == "📊 Comparaison des modèles":
    st.markdown('<div class="main-title">📊 Comparaison des Modèles</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyse comparative des performances sur le jeu de test réel</div>', unsafe_allow_html=True)

    if comp_df is not None:
        # Tableau interactif
        st.markdown('<div class="section-title">📋 Tableau comparatif complet</div>', unsafe_allow_html=True)
        st.dataframe(comp_df.set_index('Modèle'), use_container_width=True)

        # Graphes radar + barres
        metrics_plot = ['Accuracy','Precision','Recall','F1-Score','ROC-AUC','PR-AUC']
        models_names = comp_df['Modèle'].tolist()
        model_colors = ['#4C9BE8','#2ECC71','#E67E22','#9B59B6']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">🕸️ Radar des performances</div>', unsafe_allow_html=True)
            fig_radar = go.Figure()
            for i, (_, row) in enumerate(comp_df.iterrows()):
                vals = [float(row[m]) for m in metrics_plot]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=metrics_plot + [metrics_plot[0]],
                    name=row['Modèle'], line_color=model_colors[i], fill='toself',
                    fillcolor=model_colors[i], opacity=0.15
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                height=400, showlegend=True,
                legend=dict(orientation='h', y=-0.15)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">📊 F1 & Recall comparatifs</div>', unsafe_allow_html=True)
            fig_comp = go.Figure()
            for metric, color in [('F1-Score','#2ECC71'),('Recall','#E85D5D')]:
                fig_comp.add_trace(go.Bar(
                    name=metric,
                    x=comp_df['Modèle'],
                    y=[float(v) for v in comp_df[metric]],
                    marker_color=color, opacity=0.8,
                    text=[f'{float(v):.3f}' for v in comp_df[metric]],
                    textposition='outside'
                ))
            fig_comp.update_layout(
                barmode='group', height=400,
                yaxis=dict(range=[0, 1.15]),
                xaxis_tickangle=-20,
                legend=dict(orientation='h', y=1.1)
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        # Analyse critique
        st.markdown('<div class="section-title">💡 Analyse critique & justification du choix</div>', unsafe_allow_html=True)
        st.markdown("""
| Critère | Logistic Regression | Random Forest ⭐ | Gradient Boosting | MLP |
|---|---|---|---|---|
| **Performance** | Bonne | **Excellente** | Excellente | Très bonne |
| **Recall** | 0.909 | **0.993** | 0.993 | 0.972 |
| **Vitesse train** | ⚡ 0.1s | 🟢 9.7s | 🔴 30.2s | 🟢 9.4s |
| **Interprétabilité** | ✅ Haute | ✅ Haute | 🟡 Moyenne | ❌ Faible |
| **Stabilité CV** | 0.925±0.003 | **0.998±0.001** | 0.999±0.001 | 0.995±0.001 |
| **Déploiement** | ✅ Simple | ✅ Simple | 🟡 Modéré | 🟡 Modéré |

**→ Random Forest sélectionné** : meilleur compromis performance / vitesse / interprétabilité.
Gradient Boosting a des performances identiques mais est **3x plus lent** à entraîner, sans gain mesurable.
Le MLP n'apporte aucune plus-value sur ce jeu de données tabulaire structuré.
        """)
    else:
        st.error("Artefacts non disponibles. Lancez d'abord modeling.py")


# ─────────────────────────────────────────────────────────────
# PAGE 4 — IMPORTANCE DES VARIABLES
# ─────────────────────────────────────────────────────────────
elif page == "🔍 Importance des variables":
    st.markdown('<div class="main-title">🔍 Importance des Variables</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Quels signaux capteurs influencent le plus les prédictions ?</div>', unsafe_allow_html=True)

    if feat_imp is not None:
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown('<div class="section-title">📊 Permutation Importance (Random Forest)</div>', unsafe_allow_html=True)
            n_show = st.slider("Nombre de variables à afficher", 5, len(feat_imp), 10)
            top_df = feat_imp.head(n_show).copy()
            top_df['color'] = top_df['feature'].apply(
                lambda x: '🔴 Feature engineered' if x in ['temp_relative','vibration_per_rpm','maintenance_stress']
                          else ('🟠 Capteur physique' if x in ['vibration_rms','temperature_motor','rpm','pressure_level','current_phase_avg','ambient_temp']
                                else '🔵 Autre'))

            fig_imp = px.bar(top_df, x='importance', y='feature', orientation='h',
                             error_x='std', color='color',
                             color_discrete_map={
                                 '🔴 Feature engineered': '#E85D5D',
                                 '🟠 Capteur physique':   '#E67E22',
                                 '🔵 Autre':              '#4C9BE8'
                             },
                             labels={'importance': 'Chute de F1 (permutation)', 'feature': 'Variable'},
                             title="Permutation Importance — Plus la valeur est élevée, plus la variable est critique")
            fig_imp.update_layout(height=450, yaxis={'categoryorder':'total ascending'},
                                  legend_title="Type")
            st.plotly_chart(fig_imp, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">💡 Interprétation métier</div>', unsafe_allow_html=True)
            interpretations = {
                "rul_hours":           "⏱️ **Durée de vie restante** — Signal le plus fort. Plus elle est faible, plus le risque est élevé. Planifier la maintenance avant 20h restantes.",
                "vibration_per_rpm":   "🔴 **Vibration/RPM** *(engineered)* — Vibration anormale par rapport à la vitesse. Signature d'usure mécanique.",
                "temperature_motor":   "🌡️ **Température moteur** — Surchauffe = signe de défaillance. Seuil d'alerte : >70°C.",
                "rpm":                 "⚙️ **RPM** — Vitesse de rotation. Variations anormales corrélées aux pannes.",
                "temp_relative":       "🔴 **Temp. relative** *(engineered)* — Écart entre moteur et ambiant. Capte la surchauffe indépendamment de la saison.",
                "current_phase_avg":   "⚡ **Courant moyen** — Surconsommation électrique = signe de surcharge ou court-circuit.",
                "hours_since_maintenance": "🔧 **Heures sans maintenance** — Risque croissant avec le temps. Maintenance recommandée avant 400h.",
            }
            for feat, interp in interpretations.items():
                if feat in feat_imp['feature'].values:
                    rank = feat_imp[feat_imp['feature']==feat].index[0] + 1
                    with st.expander(f"#{rank} — {feat}"):
                        st.markdown(interp)

            st.markdown('<div class="section-title">📌 Conclusion pour la maintenance</div>', unsafe_allow_html=True)
            st.info("""
**Surveiller en priorité :**
1. `rul_hours` < 20h → intervention immédiate
2. `vibration_per_rpm` > seuil → inspection mécanique
3. `temperature_motor` > 70°C → vérification refroidissement
4. `hours_since_maintenance` > 400h → maintenance préventive
            """)
    else:
        st.error("Artefacts non disponibles.")


# ─────────────────────────────────────────────────────────────
# PAGE 5 — API & INFOS SYSTÈME
# ─────────────────────────────────────────────────────────────
elif page == "⚙️ API & Infos système":
    st.markdown('<div class="main-title">⚙️ API & Infos Système</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">🌐 Statut de l\'API</div>', unsafe_allow_html=True)
        if st.button("🔄 Vérifier le statut", use_container_width=True):
            try:
                r = requests.get(f"{API_URL}/health", timeout=3)
                st.json(r.json())
            except:
                st.error("API non joignable")

        st.markdown('<div class="section-title">📋 Exemple de requête</div>', unsafe_allow_html=True)
        example = {
            "vibration_rms": 3.5, "temperature_motor": 72.0,
            "current_phase_avg": 7.1, "pressure_level": 26.0, "rpm": 1450.0,
            "hours_since_maintenance": 380.0, "ambient_temp": 21.0,
            "rul_hours": 18.0, "machine_type": "Compressor", "operating_mode": "peak"
        }
        st.code(f"""# POST /predict
import requests
response = requests.post(
    "http://localhost:8000/predict",
    json={json.dumps(example, indent=4)}
)
print(response.json())""", language='python')

    with col2:
        st.markdown('<div class="section-title">📊 Informations sur le modèle</div>', unsafe_allow_html=True)
        try:
            r = requests.get(f"{API_URL}/model-info", timeout=3)
            if r.status_code == 200:
                info = r.json()
                st.metric("Modèle", info['model_name'])
                st.metric("ROC-AUC",  f"{info['test_metrics']['roc_auc']:.4f}")
                st.metric("Recall",    f"{info['test_metrics']['recall']:.4f}")
                st.metric("F1-Score",  f"{info['test_metrics']['f1']:.4f}")
                st.metric("Seuil production", f"{info['threshold']:.2f}")
                st.metric("Features", info['n_features'])
        except:
            if results:
                st.metric("Modèle", results.get('best_model_name','N/A'))
                metrics = results.get('test_metrics',{}).get(results.get('best_model_name',''),{})
                if metrics:
                    st.metric("ROC-AUC",  f"{metrics.get('roc_auc',0):.4f}")
                    st.metric("F1-Score",  f"{metrics.get('f1',0):.4f}")
                    st.metric("Recall",    f"{metrics.get('recall',0):.4f}")

        st.markdown('<div class="section-title">🗂️ Architecture du pipeline</div>', unsafe_allow_html=True)
        st.code("""
📁 projet_maintenance/
├── data_preparation.py   # Pipeline sklearn
├── modeling.py           # Entraînement 4 modèles
├── app/
│   ├── api.py            # FastAPI REST
│   └── dashboard.py      # Streamlit
└── artefacts/
    ├── preprocessor.pkl  # Pipeline fit
    ├── best_model.pkl    # Random Forest
    └── results.json      # Métriques
        """, language='text')
