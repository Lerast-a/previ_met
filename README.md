# ClimaCast — Application de Prévision Météo
### Projet 8 · B3 IABD · Collège de Paris Supérieur · 2025/2026

---

##  Structure du projet

```
meteo_app/
│
├── app.py                  ← Application Streamlit principale
├── requirements.txt        ← Dépendances Python
├── README.md               ← Ce fichier
│
├── model_lstm_v2.keras     ← copier depuis votre notebook
├── model_gru_v2.keras      ← copier depuis votre notebook
└── scaler_jena.pkl         ← À copier depuis votre notebook
```

---

##  Installation rapide

### Étape 1 — Créer un environnement virtuel (recommandé)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Étape 2 — Installer les dépendances
```bash
pip install -r requirements.txt
```

### Étape 3 — Copier vos modèles entraînés
Après avoir exécuté votre notebook `Groupe08_CC_IntroDL_v2.ipynb`,
copiez ces 3 fichiers dans le dossier `meteo_app/` :

```
model_lstm_v2.keras
model_gru_v2.keras
scaler_jena.pkl
```

### Étape 4 — Lancer l'application
```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur à :
**http://localhost:8501**

---

##  Fonctionnalités

| Onglet | Description |
|--------|-------------|
|  Prédiction manuelle | Entrez les conditions météo actuelles → prédiction T+1h |
|  Prédiction depuis CSV | Chargez un fichier CSV Jena Climate → prédiction automatique |
|  Comparaison LSTM vs GRU | Analyse architecturale + courbes d'apprentissage comparatives |

---

##  Dépannage

** "model_lstm_v2.keras introuvable"**
→ Vérifiez que les fichiers `.keras` sont bien dans le même dossier que `app.py`

** "ModuleNotFoundError: streamlit"**
→ Relancez `pip install -r requirements.txt`

** L'application ne s'ouvre pas**
→ Ouvrez manuellement http://localhost:8501 dans votre navigateur

** Erreur TensorFlow sur Windows**
→ Essayez : `pip install tensorflow-cpu` à la place de `tensorflow`

---

##  Pour générer vos propres données de test CSV

```python
import pandas as pd
import numpy as np

# Simuler 50 heures de données météo
np.random.seed(42)
n = 50
df_test = pd.DataFrame({
    'T (degC)':       np.random.uniform(5, 20, n),
    'p (mbar)':       np.random.uniform(995, 1025, n),
    'rh (%)':         np.random.uniform(60, 90, n),
    'wv (m/s)':       np.random.uniform(0, 8, n),
    'VPmax (mbar)':   np.random.uniform(8, 25, n),
    'sh (g/kg)':      np.random.uniform(4, 10, n),
    'rho (g/m**3)':   np.random.uniform(1200, 1280, n),
})
df_test.to_csv('test_meteo.csv', index=False)
print(" Fichier test_meteo.csv généré")
```

---
