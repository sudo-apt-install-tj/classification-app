# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(page_title="Star Classification", layout="wide")
st.title("⭐ Star Classification")

st.markdown("""
Upload a CSV with columns **Temperature_K, Luminosity_Lo, Radius_Ro, Absolute_Magnitude, Star_Color, Spectral_Class**  
and get back “Star_Type” predictions.
""")

uploaded = st.sidebar.file_uploader(
    "Upload your CSV for prediction", type=["csv"]
)

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("#### Preview of your data", df.head())

    df = df.rename(columns={
        'Temperature_K':'temp',
        'Luminosity_Lo':'lumins',
        'Radius_Ro':'radius',
        'Absolute_Magnitude':'absmag',
        'Star_Color':'colour',
        'Spectral_Class':'class'
    })
    df.loc[df.lumins <= 0, 'lumins'] = 1e-5
    df.loc[df.radius <= 0, 'radius'] = 1e-5

    df['stefan_law'] = (df.radius**2) * (df.temp**4)

    scaler: StandardScaler    = joblib.load("scaler.pkl")
    le_colour: LabelEncoder   = joblib.load("le_colour.pkl")
    le_class: LabelEncoder    = joblib.load("le_class.pkl")
    model = joblib.load("model.pkl")

    df['colour'] = le_colour.transform(df['colour'])
    df['class']  = le_class.transform(df['class'])

    X = df[["temp","radius","absmag","lumins","stefan_law","colour","class"]]
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    type_map = {
        0:'Red Dwarf', 1:'Brown Dwarf', 2:'White Dwarf',
        3:'Main Sequence', 4:'Supergiant', 5:'Hypergiant'
    }
    df['Predicted_Type'] = [type_map[p] for p in preds]

    st.write("#### Predictions", df[['Predicted_Type']].head())

    csv = df[['Predicted_Type']].to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Predictions as CSV",
        data=csv,
        file_name="star_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file in the sidebar to get started.")
