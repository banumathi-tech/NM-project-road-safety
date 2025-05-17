import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load and clean data
df = pd.read_csv("C:/Users/anbus/OneDrive/Desktop/nmproject/datasets/archive.csv")
df.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Target selection
target = 'Accident_Severity' if 'Accident_Severity' in df.columns else df.columns[-1]
X = df.drop(columns=[target])
y = df[target]

# Train-test split and model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'model/model.pkl')
joblib.dump(label_encoders, 'model/encoders.pkl')
print("Model and encoders saved.")