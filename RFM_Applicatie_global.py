import streamlit as st
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import urllib.request
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math

# Cache de gegevens en het model om laadtijd te verminderen
@st.cache_data
def load_data():
    data_path = 'https://raw.githubusercontent.com/BasDamen/streamlit-inzet-ambulance-AI-app/main/bronbestand_AI_model.xlsx'
    return pd.read_excel(data_path)

model_url = 'https://raw.githubusercontent.com/BasDamen/streamlit-inzet-ambulance-AI-app/main/random_forest_model.pkl'
model_path = 'random_forest_model.pkl'

# Cache de functie voor het laden van het model
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# Download het bestand naar de lokale map
urllib.request.urlretrieve(model_url, model_path)

# Laad de gegevens en het model
aggregated_data_model_gebruiken = load_data()
RFM = load_model(model_path)


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


# Configureer de LIME explainer en geef alle features als categorisch op
explainer = LimeTabularExplainer(
    training_data=data_model_gebruiken_heatmap[feature_selection].values,
    feature_names=feature_selection,
    mode='regression',
    categorical_features=list(range(len(feature_selection)))  # Alle features zijn categorisch
)


# Bouw een dynamische bezettingstijden mapping
bezettingstijden_dynamic = aggregated_data_model_gebruiken.groupby(
    ['Dagdeel', 'Inzetlocatie RAV-regio', 'Urgentie MKA']
)['gemiddelde_bezettijd'].mean().to_dict()

# Hoofd titel van de applicatie
st.title("AI Voorspellingsapplicatie - Ambulance Inzet")

# Sidebar menu met knoppen
with st.sidebar:
    st.title("Navigatie")
    page = st.radio("Selecteer Pagina", ["Informatie", "Voorspelling"])

# Homepagina
if page == "Informatie":
    st.subheader("Welkom in de applicatie voor de voorspelling van de drukte op basis van externe factoren")
    st.write(""" 
    Deze applicatie voorspelt het aantal ambulanceritten die gereden moeten worden per dag aan de hand van de ingevoerde waardes. De voorspellingen worden gemaakt met een foutmarge van 5 ambulanceritten. Dit betekent dat er een verschil is van ongeveer 5 ritten boven of onder de voorspelde waarde.
    Wanneer een voorspelling is gemaakt met de ingevoerde waardes, kun je de toelichting van de voorspelling vinden in de Lime Explainer.
             
    Daarnaast is er nog een knop die berekent hoeveel MCA of ALS ambulances nodig zijn op basis van het aantal voorspelde ambulanceritten. Hier wordt rekening gehouden met de inzetbaarheid van de ambulances en de gemiddelde bezettingstijd. *Deze functie moet worden bijgewerkt indien er verandering plaatsvinden in de inzetbaarheid van een ambulances*.

    **Wat is de Lime Explainer?**
             
    De Lime Explainer is een tool die uitlegt waarom een machine learning model een bepaalde voorspelling heeft gemaakt. 
    Het geeft inzicht in welke factoren het meest invloed hebben gehad op het resultaat, zodat je de voorspelling beter kunt begrijpen.
    
    *LIME-visualisatie:*

    De grafiek toont de invloed van verschillende featurewaarden op de voorspelling, waarbij de horizontale as de invloed van elk kenmerk aangeeft. Hoe verder een feature naar rechts of links is, hoe groter de invloed op de voorspelling (positief of negatief).
    De verticale as toont de verschillende waardes (features) van de voorspelling.
    - Een waarde die van invloed is op de voorspelling wordt getoond met '= 0' of '= 1'. Bij een waarde 0 betekent het dat de waarde niet aanwezig is en bij een 1 wel. 
        - Als bijvoorbeeld Dagdeel_Nacht en Dagdeel_Dag beide de waarde 0 hebben, dan is Dagdeel_Avond de waarde 1. Deze toont de grafiek alleen niet.    
    
    Voor meer informatie, bekijk de volgende video: [Lime Explainer Explained!](https://www.youtube.com/watch?v=d6j6bofhj2M&list=PLV8yxwGOxvvovp-j6ztxhF3QcKXT6vORU&index=3)
    """)
   
# Voorspelling pagina
elif page == "Voorspelling":
    st.subheader("Maak een voorspelling voor het aantal ambulanceritten en de benodigde ambulances")

    # Maak keuzemenu's voor de parameters (meervoudige selectie)
    dagdeel = st.multiselect(
        "Kies Dagdeel/Dagdelen", 
        ["Dag", "Avond", "Nacht"], 
        default=[]
    )

    urgenties = st.multiselect(
        "Kies Urgentie(s)", 
        ["A1", "A2", "B"], 
        default=[]
    )

    inzetlocaties = st.multiselect(
        "Kies RAV-regio('s)", 
        ["Regio 23", "Regio 24"], 
        default=[]
    )

    # Voeg de selectbox voor Weekend toe
    weekend = st.selectbox(
        "Is het Weekend?", 
        ["Ja", "Nee"],
        index=1  # Default "Nee"
    )

    # Check if any input changed and reset session state
    input_hash = hash((tuple(dagdeel), tuple(urgenties), tuple(inzetlocaties), weekend))
    if "input_hash" in st.session_state and st.session_state.input_hash != input_hash:
        # Reset the session state when the input changes
        st.session_state.lime_explanations = []
        st.session_state.total_prediction = None

    # Save the current input hash in session state
    st.session_state.input_hash = input_hash

    # Controleer of alle velden zijn ingevuld
    is_valid_input = bool(dagdeel) and bool(urgenties) and bool(inzetlocaties)
    make_prediction_disabled = not is_valid_input  # Disable de knop als de invoer niet geldig is

    # Voorspelling uitvoeren bij het klikken op de knop
    if st.button("Maak voorspelling", disabled=make_prediction_disabled):
        total_prediction = 0
        lime_explanations = []

        # Zet Weekend om naar binaire waarde
        weekend_value = 1 if weekend == "Ja" else 0

        # Loop door de geselecteerde dagdelen, urgenties en regio's om voorspellingen en uitleg te maken
        for dag in dagdeel:
            for urgentie in urgenties:
                for regio in inzetlocaties:
                    data = {
                        "Inzetlocatie RAV-regio_RAV Zuid Limburg (24)" if regio == "Regio 24" else "Inzetlocatie RAV-regio_RAV Noord- en Midden Limburg": [1 if regio == "Regio 24" else 0],
                        "Dagdeel_Nacht": [1 if dag == "Nacht" else 0],
                        "Dagdeel_Dag": [1 if dag == "Dag" else 0],
                        "Urgentie MKA_A2": [1 if urgentie == "A2" else 0],
                        "Urgentie MKA_B": [1 if urgentie == "B" else 0],
                        "Weekend": [weekend_value]  # Gebruik de binaire waarde voor Weekend
                    }

                    X_inference = pd.DataFrame(data)
                    prediction = RFM.predict(X_inference)
                    rounded_prediction = math.ceil(prediction[0])
                    total_prediction += rounded_prediction

                    exp = explainer.explain_instance(X_inference.values[0], RFM.predict, num_features=len(feature_selection))
                    lime_explanations.append({
                        "Dagdeel": dag,
                        "Urgentie": urgentie,
                        "Regio": regio,
                        "Prediction": rounded_prediction,
                        "Explanation": exp
                    })

        # Sla de uitleg en voorspelling op in de sessie om ze later te tonen
        st.session_state.lime_explanations = lime_explanations
        st.session_state.total_prediction = math.ceil(total_prediction)

        # Toon de totale voorspelling met grotere tekst
        st.markdown(f"<h4>Totale voorspelling aantal ambulanceritten (afgerond naar boven): {st.session_state.total_prediction}</h4>", unsafe_allow_html=True)

        
        # Witregel toevoegen tussen de voorspelling en de uitleg
        st.write("\n")

        # Gebruik een expander om de gedetailleerde uitleg weer te geven
        with st.expander("Bekijk de gedetailleerde uitleg van de voorspelling(en) met LIME"):
            if 'lime_explanations' in st.session_state and st.session_state.lime_explanations:
                for exp_data in st.session_state.lime_explanations:
                    st.write(f"Voor de waardes: {exp_data['Dagdeel']} - {exp_data['Urgentie']} - {exp_data['Regio']}")
                    st.write(f"Voorspelling ambulanceritten: {exp_data['Prediction']}")

                    # Genereer de LIME-explainer figuur
                    fig = exp_data["Explanation"].as_pyplot_figure()

                    # Pas de assenlabels aan
                    fig.axes[0].set_xlabel('Invloed op Voorspelling', fontsize=14)  # X-as label
                    fig.axes[0].set_ylabel('Waardes (Features)', fontsize=14)   # Y-as label

                    # Pas de titel aan
                    fig.suptitle('Verklaring van de Voorspelling door LIME', fontsize=16)

                    # Toon de aangepaste figuur
                    st.pyplot(fig)

                    st.write("---")
            else:
                st.write("Voorspelling is nog niet gemaakt. Klik op 'Maak voorspelling' om eerst een voorspelling te genereren.")

    # Controleer of voorspellingen beschikbaar zijn
    if "lime_explanations" in st.session_state and st.session_state.lime_explanations:
        if st.button("Bereken benodigde ambulances"):
            # Maak een lege dictionary om de benodigde ambulances per dagdeel op te slaan
            ambulances_per_dagdeel = {}

            # Loop door de LIME verklaringen en bereken het aantal benodigde ambulances per dagdeel
            for exp_data in st.session_state.lime_explanations:
                prediction = exp_data['Prediction']
                dagdeel = exp_data['Dagdeel']
                urgentie = exp_data['Urgentie']
                regio = exp_data['Regio']

                # Controleer of de sleutels bestaan in de bezettingstijden
                key = (dagdeel,(regio.replace("Regio 24", "RAV Zuid Limburg (24)").replace("Regio 23", "RAV Noord- en Midden Limburg (23)")), urgentie)
                try:
                    bezettingstijd = bezettingstijden_dynamic[key]
                except KeyError as e:
                    st.error(f"Fout bij het ophalen van bezettingstijd: {e}")
                    st.write(f"Controleer of de invoerwaarden correct zijn: dagdeel={dagdeel}, regio={regio}, urgentie={urgentie}")
                    bezettingstijd = 0  # Gebruik standaardwaarde om crash te voorkomen


                # Bereken de bezettingstijd per dagdeel
                if dagdeel not in ambulances_per_dagdeel:
                    ambulances_per_dagdeel[dagdeel] = {'A': 0, 'B': 0}  # Initialiseer het dagdeel in de dictionary

                if urgentie == 'B':  # B-urgentie is MCA
                    ambulances_per_dagdeel[dagdeel]['B'] += prediction * bezettingstijd
                else:  # A-urgentie is ALS
                    ambulances_per_dagdeel[dagdeel]['A'] += prediction * bezettingstijd

            # Bereken het aantal benodigde ambulances per dagdeel
            for dagdeel, bezetting in ambulances_per_dagdeel.items():
                aantal_ambulances_B = math.ceil(bezetting['B'] / 336) # beschikbaarheid van 70% voor 8-uurige MCA dienst (480 minuten)
                aantal_ambulances_A = math.ceil(bezetting['A'] / 240) # beschikbaarheid van 50% voor 8-uurige ALS dienst (480 minuten)

                # Toon de resultaten per dagdeel
                st.write(f"Aantal benodigde MCA-ambulances voor B-urgentie in dagdeel {dagdeel}: {aantal_ambulances_B}")
                st.write(f"Aantal benodigde ALS-ambulances voor A-urgentie in dagdeel {dagdeel}: {aantal_ambulances_A}")