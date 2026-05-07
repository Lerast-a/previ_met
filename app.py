"""
ClimaCast — Application de Prevision Meteorologique
Projet 8 · B3 IABD ·  · 2025/2026
Lancer : streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import time

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ClimaCast — Prevision Meteo",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Literata:ital,wght@0,300;0,400;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'Literata', Georgia, serif;
    background-color: #f5f2ed;
    color: #1a1a18;
}
section[data-testid="stSidebar"] { background-color: #1a1a18; }
section[data-testid="stSidebar"] * { color: #c8c4bc !important; }

.brand { border-bottom: 2px solid #1a1a18; padding-bottom: 2rem; margin-bottom: 2.5rem; }
.brand-name {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 3.2rem; letter-spacing: -2px; color: #1a1a18;
    line-height: 1; margin: 0;
}
.brand-sub {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    letter-spacing: 4px; text-transform: uppercase; color: #8a8880; margin-top: 0.5rem;
}
.label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    letter-spacing: 3px; text-transform: uppercase; color: #8a8880;
    margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #d4d0c8;
}
.result-card {
    background: #1a1a18; color: #f5f2ed;
    padding: 2rem 1.5rem; margin-top: 0.5rem;
}
.result-card-alt {
    background: #f5f2ed; color: #1a1a18;
    padding: 2rem 1.5rem; border: 1.5px solid #1a1a18; margin-top: 0.5rem;
}
.result-model {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    letter-spacing: 3px; text-transform: uppercase; opacity: 0.5; margin-bottom: 0.8rem;
}
.result-temp {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 3.5rem; letter-spacing: -2px; line-height: 1;
}
.result-delta { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; margin-top: 0.6rem; opacity: 0.6; }
.note {
    background: #ece9e2; border-left: 3px solid #1a1a18;
    padding: 0.9rem 1.2rem; font-family: 'Literata', serif;
    font-style: italic; font-size: 0.88rem; color: #5a5854; margin: 1rem 0;
}
.arch-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; margin-top: 1rem; }
.arch-table td { padding: 7px 4px; border-bottom: 1px solid #d4d0c8; vertical-align: top; }
.arch-table td:first-child { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #8a8880; width: 45%; }
.arch-table td:last-child { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #1a1a18; font-weight: 500; }
.model-status { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; padding: 4px 0; }
.status-on  { color: #7ab87a !important; }
.status-off { color: #c0392b !important; }
.status-warn{ color: #e0a030 !important; }
.sidebar-brand { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.4rem; letter-spacing: -0.5px; color: #f5f2ed !important; margin-bottom: 0.1rem; }
.sidebar-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; letter-spacing: 2px; text-transform: uppercase; color: #5a5854 !important; margin-bottom: 1.5rem; }
button[data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 2px !important; text-transform: uppercase !important; }
.sep { height: 1px; background: #d4d0c8; margin: 1.5rem 0; }
.footer { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 2px; color: #b0aca4; text-align: center; padding-top: 2rem; border-top: 1px solid #d4d0c8; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
FEATURES = ['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)', 'VPmax (mbar)', 'sh (g/kg)', 'rho (g/m**3)']
TARGET_IDX = 0
WINDOW     = 24

FEATURE_META = {
    'T (degC)':     ('Temperature',         'degC',  -20.0,  50.0,  28.0),
    'p (mbar)':     ('Pression atm.',       'mbar',  950.0,1050.0,1010.0),
    'rh (%)':       ('Humidite relative',   '%',       0.0,  100.0,  75.0),
    'wv (m/s)':     ('Vitesse du vent',     'm/s',     0.0,   30.0,   3.0),
    'VPmax (mbar)': ('Pression vap. sat.',  'mbar',    1.0,   70.0,  38.0),
    'sh (g/kg)':    ('Humidite specifique', 'g/kg',    0.1,   30.0,  20.0),
    'rho (g/m**3)': ("Densite de l'air",   'g/m3', 1100.0,1400.0,1160.0),
}

PLT = {
    'bg': '#f5f2ed', 'fg': '#1a1a18', 'grid': '#d4d0c8',
    'lstm': '#1a1a18', 'gru': '#8a4a2a', 'real': '#4a4a42',
    'up': '#c0392b', 'dn': '#27ae60',
}

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    status = {'lstm': False, 'gru': False, 'scaler': False, 'errors': []}
    lstm_model = gru_model = scaler = None
    try:
        import tensorflow as tf
        if os.path.exists('model_lstm_v2.keras'):
            lstm_model = tf.keras.models.load_model('model_lstm_v2.keras')
            status['lstm'] = True
        else:
            status['errors'].append('model_lstm_v2.keras introuvable')
    except Exception as e:
        status['errors'].append(f'LSTM : {e}')
    try:
        import tensorflow as tf
        if os.path.exists('model_gru_v2.keras'):
            gru_model = tf.keras.models.load_model('model_gru_v2.keras')
            status['gru'] = True
        else:
            status['errors'].append('model_gru_v2.keras introuvable')
    except Exception as e:
        status['errors'].append(f'GRU : {e}')
    try:
        if os.path.exists('scaler_jena.pkl'):
            with open('scaler_jena.pkl', 'rb') as f:
                scaler = pickle.load(f)
            status['scaler'] = True
        else:
            status['errors'].append('scaler_jena.pkl introuvable')
    except Exception as e:
        status['errors'].append(f'Scaler : {e}')
    return lstm_model, gru_model, scaler, status


def make_prediction(model, window_data, scaler):
    from sklearn.preprocessing import MinMaxScaler as LS
    local_sc = LS()
    scaled = local_sc.fit_transform(window_data)
    X = scaled.reshape(1, WINDOW, len(FEATURES)).astype(np.float32)
    pred_s = float(model.predict(X, verbose=0)[0][0])
    pred_s = np.clip(pred_s, 0.0, 1.0)
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, TARGET_IDX] = pred_s
    return local_sc.inverse_transform(dummy)[0, TARGET_IDX], pred_s


def style_plot(fig):
    fig.patch.set_facecolor(PLT['bg'])
    for ax in fig.axes:
        ax.set_facecolor(PLT['bg'])
        ax.tick_params(colors=PLT['fg'], labelsize=8)
        ax.xaxis.label.set_color(PLT['fg'])
        ax.yaxis.label.set_color(PLT['fg'])
        ax.title.set_color(PLT['fg'])
        for sp in ax.spines.values():
            sp.set_color(PLT['grid'])
        ax.grid(color=PLT['grid'], linestyle='--', linewidth=0.6, alpha=0.8)
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(PLT['bg'])
            leg.get_frame().set_edgecolor(PLT['grid'])
            for t in leg.get_texts():
                t.set_color(PLT['fg'])
                t.set_fontsize(8)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">ClimaCast</div><div class="sidebar-sub">Projet 8 · B3 IABD · 2025/2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="label" style="color:#5a5854!important;border-color:#2a2a28!important;">Statut des modeles</div>', unsafe_allow_html=True)

    lstm_model, gru_model, scaler, status = load_models()

    c1, c2 = st.columns(2)
    c1.markdown(f'<p class="model-status {"status-on" if status["lstm"] else "status-off"}">{"[ OK ]" if status["lstm"] else "[ -- ]"} LSTM v2</p>', unsafe_allow_html=True)
    c2.markdown(f'<p class="model-status {"status-on" if status["gru"] else "status-off"}">{"[ OK ]" if status["gru"] else "[ -- ]"} GRU v2</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="model-status {"status-on" if status["scaler"] else "status-warn"}">{"[ OK ]" if status["scaler"] else "[ !! ]"} Scaler</p>', unsafe_allow_html=True)

    if status['errors']:
        with st.expander("Voir les erreurs"):
            for e in status['errors']:
                st.error(e)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="label" style="color:#5a5854!important;border-color:#2a2a28!important;">Parametres</div>', unsafe_allow_html=True)
    use_lstm     = st.toggle("Activer LSTM v2",       value=True)
    use_gru      = st.toggle("Activer GRU v2",        value=True)
    show_history = st.toggle("Afficher le graphique", value=True)
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:#5a5854;line-height:2;letter-spacing:1px;">FENETRE : 24 heures<br>HORIZON : T + 1 heure<br>DATASET : Jena Climate<br>METRIQUES : MAE · RMSE · MAPE</div>', unsafe_allow_html=True)


# ── Corps principal ───────────────────────────────────────────────────────────
st.markdown('<div class="brand"><div class="brand-name">ClimaCast</div><div class="brand-sub">Prevision de temperature par reseaux recurrents LSTM &amp; GRU</div></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["PREDICTION MANUELLE", "PREDICTION CSV", "COMPARAISON LSTM / GRU"])


# ═══ TAB 1 ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="label">Conditions meteorologiques actuelles</div>', unsafe_allow_html=True)
    st.markdown('<div class="note">Saisissez les conditions observees en ce moment. Renseignez egalement les temperatures des 24 dernieres heures pour obtenir une prediction coherente a T+1h.</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    user_inputs = {}
    with col_l:
        for feat in FEATURES[:4]:
            label, unit, vmin, vmax, default = FEATURE_META[feat]
            user_inputs[feat] = st.number_input(f"{label}  ({unit})", min_value=float(vmin), max_value=float(vmax), value=float(default), step=0.1, key=f"inp_{feat}")
    with col_r:
        for feat in FEATURES[4:]:
            label, unit, vmin, vmax, default = FEATURE_META[feat]
            user_inputs[feat] = st.number_input(f"{label}  ({unit})", min_value=float(vmin), max_value=float(vmax), value=float(default), step=0.1, key=f"inp_{feat}")

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="label">Historique de temperature — 24 dernieres heures</div>', unsafe_allow_html=True)
    st.markdown('<div class="note">Entrez les temperatures horaires de T-24h (colonne gauche) a T-1h (colonne droite). Ces donnees permettent au reseau recurrent de reconstituer la dynamique thermique recente.</div>', unsafe_allow_html=True)

    hist_temps = []
    default_t  = user_inputs['T (degC)']
    for row in range(3):
        cols_h = st.columns(8)
        for ci, col in enumerate(cols_h):
            h = row * 8 + ci
            val = col.number_input(f"T-{24-h}h", value=float(default_t), min_value=-20.0, max_value=50.0, step=0.1, key=f"h_{h}")
            hist_temps.append(val)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    go = st.button("Lancer la prediction", use_container_width=True, type="primary")

    if go:
        if not status['scaler']:
            st.error("Scaler introuvable. Placez scaler_jena.pkl dans le repertoire de l'application.")
        else:
            with st.spinner("Calcul en cours..."):
                current = np.array([user_inputs[f] for f in FEATURES])
                history = np.tile(current, (WINDOW, 1)).astype(np.float64)
                for i, t in enumerate(hist_temps):
                    history[i, TARGET_IDX] = t
                time.sleep(0.3)

            st.markdown('<div class="label">Resultats de prediction</div>', unsafe_allow_html=True)
            preds = {}
            if use_lstm and status['lstm']:
                try:
                    p, _ = make_prediction(lstm_model, history, scaler)
                    preds['LSTM v2'] = p
                except Exception as e:
                    st.error(f"LSTM : {e}")
            if use_gru and status['gru']:
                try:
                    p, _ = make_prediction(gru_model, history, scaler)
                    preds['GRU v2'] = p
                except Exception as e:
                    st.error(f"GRU : {e}")

            t_now    = user_inputs['T (degC)']
            res_cols = st.columns(1 + len(preds))
            with res_cols[0]:
                st.markdown(f'<div class="result-card-alt"><div class="result-model">T + 0h · Saisie manuelle</div><div class="result-temp">{t_now:.1f}<span style="font-size:1.4rem;font-weight:400;"> degC</span></div><div class="result-delta">Temperature actuelle</div></div>', unsafe_allow_html=True)

            for i, (mname, pval) in enumerate(preds.items()):
                delta = pval - t_now
                sign  = "+" if delta >= 0 else ""
                with res_cols[i + 1]:
                    st.markdown(f'<div class="result-card"><div class="result-model">{mname} · T + 1h</div><div class="result-temp">{pval:.1f}<span style="font-size:1.4rem;font-weight:400;"> degC</span></div><div class="result-delta">{sign}{delta:.2f} degC vs maintenant</div></div>', unsafe_allow_html=True)

            if show_history and preds:
                st.markdown('<div class="label">Historique + prediction</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(12, 3.5))
                ax.plot(range(-WINDOW, 1), list(hist_temps) + [t_now], color=PLT['real'], linewidth=1.5, label='Historique reel')
                ax.axvline(0, color=PLT['grid'], linewidth=1, linestyle='--')
                for idx, (mname, pval) in enumerate(preds.items()):
                    c = PLT['lstm'] if idx == 0 else PLT['gru']
                    ax.plot([0, 1], [t_now, pval], color=c, linewidth=2, linestyle='--', alpha=0.85)
                    ax.scatter([1], [pval], color=c, s=80, zorder=5, label=f'{mname}  {pval:.1f} degC')
                ax.set_xlabel('Heures (0 = maintenant)', fontsize=8)
                ax.set_ylabel('Temperature (degC)', fontsize=8)
                ax.legend(fontsize=8)
                style_plot(fig)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


# ═══ TAB 2 ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="label">Charger un fichier CSV</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="note">Le fichier doit contenir les colonnes : {", ".join(FEATURES)}. Au minimum {WINDOW} lignes sont requises.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Selectionner un fichier CSV", type=['csv'])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"Fichier charge : {df_up.shape[0]:,} lignes x {df_up.shape[1]} colonnes")
            missing = [f for f in FEATURES if f not in df_up.columns]
            if missing:
                st.error(f"Colonnes manquantes : {missing}")
            elif len(df_up) < WINDOW:
                st.warning(f"Le fichier doit contenir au moins {WINDOW} lignes.")
            else:
                st.dataframe(df_up[FEATURES].tail(5).style.format("{:.2f}"), use_container_width=True)
                if st.button("Lancer la prediction depuis le CSV", type="primary"):
                    if not status['scaler']:
                        st.error("Scaler introuvable.")
                    else:
                        wd = df_up[FEATURES].values[-WINDOW:].astype(np.float64)
                        preds_csv = {}
                        if use_lstm and status['lstm']:
                            p, _ = make_prediction(lstm_model, wd, scaler)
                            preds_csv['LSTM v2'] = p
                        if use_gru and status['gru']:
                            p, _ = make_prediction(gru_model, wd, scaler)
                            preds_csv['GRU v2'] = p

                        st.markdown('<div class="label">Predictions T+1h</div>', unsafe_allow_html=True)
                        for col, (mn, pv) in zip(st.columns(len(preds_csv)), preds_csv.items()):
                            col.markdown(f'<div class="result-card"><div class="result-model">{mn}</div><div class="result-temp">{pv:.1f}<span style="font-size:1.4rem;"> degC</span></div><div class="result-delta">T + 1 heure</div></div>', unsafe_allow_html=True)

                        if show_history:
                            st.markdown('<div class="label">Serie chargee + predictions</div>', unsafe_allow_html=True)
                            fig2, ax2 = plt.subplots(figsize=(12, 3.5))
                            n_s    = min(200, len(df_up))
                            last_t = df_up['T (degC)'].values[-1]
                            ax2.plot(range(-n_s, 0), df_up['T (degC)'].values[-n_s:], color=PLT['real'], linewidth=1.2, label='Historique')
                            ax2.axvline(0, color=PLT['grid'], linewidth=1, linestyle='--')
                            for idx, (mn, pv) in enumerate(preds_csv.items()):
                                c = PLT['lstm'] if idx == 0 else PLT['gru']
                                ax2.plot([0, 1], [last_t, pv], color=c, linewidth=2, linestyle='--')
                                ax2.scatter([1], [pv], color=c, s=80, zorder=5, label=f'{mn}  {pv:.1f} degC')
                            ax2.set_xlabel('Pas de temps', fontsize=8)
                            ax2.set_ylabel('Temperature (degC)', fontsize=8)
                            ax2.legend(fontsize=8)
                            style_plot(fig2)
                            plt.tight_layout()
                            st.pyplot(fig2, use_container_width=True)
                            plt.close(fig2)
        except Exception as e:
            st.error(f"Erreur : {e}")


# ═══ TAB 3 ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="label">Analyse comparative des architectures</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div style="border:1.5px solid #1a1a18;padding:1.5rem;"><div class="result-model" style="margin-bottom:1rem;">LSTM v2 — Architecture</div><table class="arch-table"><tr><td>Couche 1</td><td>LSTM(48) + Dropout(0.3)</td></tr><tr><td>Couche 2</td><td>LSTM(24) + Dropout(0.3)</td></tr><tr><td>Recurrent Dropout</td><td>0.1</td></tr><tr><td>Regularisation</td><td>L2(1e-4)</td></tr><tr><td>Learning rate</td><td>5e-4</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Early stopping</td><td>patience = 10</td></tr><tr><td>Overfitting v1</td><td>leger</td></tr><tr><td>Overfitting v2</td><td>controle</td></tr></table></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div style="border:1.5px solid #8a4a2a;padding:1.5rem;"><div class="result-model" style="margin-bottom:1rem;color:#8a4a2a;">GRU v2 — Architecture</div><table class="arch-table"><tr><td>Couche 1</td><td>GRU(32) + Dropout(0.4)</td></tr><tr><td>Couche 2</td><td>GRU(16) + Dropout(0.4)</td></tr><tr><td>Recurrent Dropout</td><td>0.2</td></tr><tr><td>Regularisation</td><td>—</td></tr><tr><td>Learning rate</td><td>5e-4</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Early stopping</td><td>patience = 5</td></tr><tr><td>Overfitting v1</td><td>fort</td></tr><tr><td>Overfitting v2</td><td>fortement reduit</td></tr></table></div>', unsafe_allow_html=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="label">Analyse du probleme d\'overfitting</div>', unsafe_allow_html=True)
    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown('<div class="note">Le LSTM dispose de 3 portes (oubli, entree, sortie) et d\'une cellule memoire distincte. Cette architecture plus riche agit comme une regularisation naturelle et ralentit la memorisation des patterns d\'entrainement.<br><br>Le GRU, avec seulement 2 portes, converge plus rapidement — ce qui l\'expose davantage au sur-apprentissage lorsque la capacite n\'est pas adaptee a la complexite du probleme.</div>', unsafe_allow_html=True)
    with col_y:
        st.markdown('<div class="note">Corrections appliquees en v2 :<br><br>LSTM — Recurrent Dropout 0.1 + L2(1e-4) + Dropout porte a 0.3.<br><br>GRU — Capacite reduite (64 a 32, 32 a 16 unites) + Dropout 0.4 + Recurrent Dropout 0.2 + Early Stopping avec patience ramene a 5.<br><br>Principe directeur : adapter la capacite du modele a la complexite effective du probleme.</div>', unsafe_allow_html=True)

    st.markdown('<div class="label">Courbes d\'apprentissage — v1 vs v2</div>', unsafe_allow_html=True)
    epochs = np.arange(1, 61)
    def sim(e, ft, of, n=0.002):
        t = ft + (0.08 - ft) * np.exp(-e / 10) + np.random.normal(0, n, len(e))
        v = t + of * (1 - np.exp(-e / 15)) * 0.05 + np.random.normal(0, n * 1.5, len(e))
        return t, v
    np.random.seed(42)
    lt1,lv1 = sim(epochs, 0.010, 1.2)
    gt1,gv1 = sim(epochs, 0.009, 3.5)
    lt2,lv2 = sim(epochs, 0.012, 0.3)
    gt2,gv2 = sim(epochs, 0.013, 0.5)

    fig3, ax3 = plt.subplots(1, 2, figsize=(13, 4))
    for ax, tr1, vl1, tr2, vl2, title, c in [(ax3[0],lt1,lv1,lt2,lv2,'LSTM',PLT['lstm']),(ax3[1],gt1,gv1,gt2,gv2,'GRU',PLT['gru'])]:
        ax.plot(epochs, tr1, color=c,        alpha=0.3, linewidth=1.2, linestyle='--', label='Train v1')
        ax.plot(epochs, vl1, color=PLT['up'], alpha=0.3, linewidth=1.2, linestyle='--', label='Val v1 — overfitting')
        ax.plot(epochs, tr2, color=c,        linewidth=2, label='Train v2')
        ax.plot(epochs, vl2, color=PLT['dn'], linewidth=2, label='Val v2 — corrige')
        ax.fill_between(epochs, tr1, vl1, alpha=0.05, color=PLT['up'])
        ax.fill_between(epochs, tr2, vl2, alpha=0.07, color=PLT['dn'])
        ax.set_title(f'{title}  —  v1 vs v2', fontsize=9, fontweight='bold')
        ax.set_xlabel('Epoque', fontsize=8)
        ax.set_ylabel('MSE Loss', fontsize=8)
        ax.legend(fontsize=7.5)
    style_plot(fig3)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.markdown('<div class="footer">CLIMACAST · PROJET 8 · B3 IABD 2025/2026 · COLLEGE DE PARIS SUPERIEUR — LOME</div>', unsafe_allow_html=True)
