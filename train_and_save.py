# train_and_save.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("train_test_dataset.csv")
df.drop(columns='S.No.',inplace=True)
df = df.rename(columns={
    'Temperature_K':    'temp',
    'Luminosity_Lo':    'lumins',
    'Radius_Ro':        'radius',
    'Absolute_Magnitude':'absmag',
    'Star_Color':       'colour',
    'Spectral_Class':   'class',
    'Star_Type':        'type'
})

df.loc[df.lumins <= 0, 'lumins']   = 1e-5
df.loc[df.radius <= 0, 'radius']   = 1e-5
df['stefan_law'] = (df.radius ** 2) * (df.temp ** 4)

le_colour = LabelEncoder().fit(df['colour'])
le_class  = LabelEncoder().fit(df['class'])
df['colour'] = le_colour.transform(df['colour'])
df['class']  = le_class.transform(df['class'])

X = df.drop(columns=['type'])
y = df['type']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

param_grid = {
    'n_estimators':    [100, 200],
    'max_depth':       [None, 10, 20],
    'min_samples_leaf':[1, 2]
}
rfc = RandomForestClassifier(random_state=42)
rfc_grid = GridSearchCV(rfc, param_grid, cv=5, scoring='accuracy')
rfc_grid.fit(X_train_scaled, y_train)
best_model = rfc_grid.best_estimator_

print("Best RFC params:", rfc_grid.best_params_)

joblib.dump(scaler,       "scaler.pkl")
joblib.dump(le_colour,    "le_colour.pkl")
joblib.dump(le_class,     "le_class.pkl")
joblib.dump(best_model,   "model.pkl")

print("Preprocessors and model saved to disk.")
