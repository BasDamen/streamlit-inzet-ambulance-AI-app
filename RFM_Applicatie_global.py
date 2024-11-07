import streamlit as st
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import urllib.request

# Bestanden ophalen vanaf GitHub via raw URL
data_url = 'https://raw.githubusercontent.com/BasDamen/streamlit-inzet-ambulance-AI-app/main/bronbestand_AI_model.xlsx'
aggregated_data_model_gebruiken = pd.read_excel(data_url)

# Data preprocessing
# One-hot encoding voor specifieke kolommen
one_hot_columns = ['Soort_dag', 'Feestdag', 'Feestdag_specifiek', 
                   "Seizoen", "Maand", "Dag_van_de_week", 
                   "Dagdeel", 'Inzetlocatie RAV-regio', "Urgentie MKA", 'Vakantie']
data_model_gebruiken_heatmap = pd.get_dummies(aggregated_data_model_gebruiken, columns=one_hot_columns)

# Feature selectie
feature_selection = ['Inzetlocatie RAV-regio_RAV Zuid Limburg (24)', 
                    "Dagdeel_Nacht", "Dagdeel_Dag", 
                    "Urgentie MKA_A2", "Urgentie MKA_B", "Weekend"]

# Schaal de data met RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(data_model_gebruiken_heatmap[feature_selection])

# Download het modelbestand van GitHub
model_url = 'https://raw.githubusercontent.com/BasDamen/streamlit-inzet-ambulance-AI-app/main/random_forest_model.pkl'
model_path = 'random_forest_model.pkl'  #  opslaglocatie

# Download het bestand naar de lokale map
urllib.request.urlretrieve(model_url, model_path)

# Laad het model
RFM = joblib.load(model_path)

# Stel de feature-namen in
feature_names = ["Inzetlocatie RAV-regio_RAV Zuid Limburg (24)", "Dagdeel_Nacht", "Dagdeel_Dag", "Urgentie MKA_A2", "Urgentie MKA_B", "Weekend"]

# Configureer de LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_scaled,
    feature_names=feature_selection,
    mode='regression'
)

# Titel van de applicatie
st.title("AI Voorspellingsapplicatie - Ambulance Inzet")

# Maak invoervelden voor de parameters
inzetlocatie = st.selectbox("Inzetlocatie RAV-regio_RAV Zuid Limburg (24)", [0, 1])
dagdeel_nacht = st.selectbox("Dagdeel_Nacht", [0, 1])
dagdeel_dag = st.selectbox("Dagdeel_Dag", [0, 1])
urgentie_a2 = st.selectbox("Urgentie MKA_A2", [0, 1])
urgentie_b = st.selectbox("Urgentie MKA_B", [0, 1])
weekend = st.selectbox("Weekend", [0, 1])

# Voorspelling uitvoeren bij het klikken op de knop
if st.button("Maak voorspelling"):
    # Maak een DataFrame van de invoer
    data = {
        "Inzetlocatie RAV-regio_RAV Zuid Limburg (24)": [inzetlocatie],
        "Dagdeel_Nacht": [dagdeel_nacht],
        "Dagdeel_Dag": [dagdeel_dag],
        "Urgentie MKA_A2": [urgentie_a2],
        "Urgentie MKA_B": [urgentie_b],
        "Weekend": [weekend]
    }
    X_inference = pd.DataFrame(data)

    # Schaal de invoerdata
    X_inference_scaled = scaler.transform(X_inference)

    # Voorspellingen maken
    predictions = RFM.predict(X_inference_scaled)

    # Toon het resultaat, afgerond op 2 decimalen
    st.write("Voorspelling aantal ambulanceritten:", round(predictions[0], 2))

    # LIME-explanation genereren
    exp = explainer.explain_instance(X_inference_scaled[0], RFM.predict, num_features=len(feature_selection))

    # LIME-explanation weergeven
    st.subheader("LIME Explanation - Inzicht in de voorspelling")
    for feature, contribution in exp.as_list():
        st.write(f"{feature}: {contribution:.4f}")

    # Plot de LIME uitleg
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)
