# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder  # for type hints only

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
    df.drop(columns='S.No.',inplace=True)
    st.write("#### Preview of your data", df.head())

    # — rename to match training —
    df = df.rename(columns={
        'Temperature_K':    'temp',
        'Luminosity_Lo':    'lumins',
        'Radius_Ro':        'radius',
        'Absolute_Magnitude':'absmag',
        'Star_Color':       'colour',
        'Spectral_Class':   'class'
    })

    # — clean zeros & feature engineer —
    df.loc[df.lumins <= 0, 'lumins'] = 1e-5
    df.loc[df.radius <= 0, 'radius'] = 1e-5
    df['stefan_law'] = (df.radius**2) * (df.temp**4)

    # — load your pickles —
    scaler: StandardScaler    = joblib.load("scaler.pkl")
    le_colour: LabelEncoder   = joblib.load("le_colour.pkl")
    le_class: LabelEncoder    = joblib.load("le_class.pkl")
    model = joblib.load("model.pkl")

    # — OLD: this would error on unseen labels —
    # df['colour'] = le_colour.transform(df['colour'])
    # df['class']  = le_class.transform(df['class'])

    # — NEW: map known → original index, unseen → new bucket index —
    colour_map = {lab: idx for idx, lab in enumerate(le_colour.classes_)}
    default_colour_idx = len(colour_map)
    df['colour_code'] = df['colour'].map(lambda x: colour_map.get(x, default_colour_idx))

    class_map = {lab: idx for idx, lab in enumerate(le_class.classes_)}
    default_class_idx = len(class_map)
    df['class_code'] = df['class'].map(lambda x: class_map.get(x, default_class_idx))

    # — prepare feature matrix with your new code columns —
    X = df[[
        "temp",
        "radius",
        "absmag",
        "lumins",
        "stefan_law",
        "colour_code",
        "class_code"
    ]]
    X_scaled = scaler.transform(X.to_numpy())

    # — predict —
    preds = model.predict(X_scaled)

    type_map = {
        0: 'Red Dwarf',
        1: 'Brown Dwarf',
        2: 'White Dwarf',
        3: 'Main Sequence',
        4: 'Supergiant',
        5: 'Hypergiant'
    }
    df['Predicted_Type'] = [type_map[p] for p in preds]

    st.write("#### Predictions", df[['Predicted_Type']].head())

    # — download button —
    csv = df[['Predicted_Type']].to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Predictions as CSV",
        data=csv,
        file_name="star_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file in the sidebar to get started.")
