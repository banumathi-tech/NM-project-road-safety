from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load('model/model.pkl')
label_encoders = joblib.load('model/encoders.pkl')

# Prediction route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Collect input values
        user_input = {
            'traffic_control_device': request.form['traffic_control_device'],
            'weather_condition': request.form['weather_condition'],
            'lighting_condition': request.form['lighting_condition'],
            'first_crash_type': request.form['first_crash_type'],
            'trafficway_type': request.form['trafficway_type'],
            'alignment': request.form['alignment'],
            'roadway_surface_cond': request.form['roadway_surface_cond'],
            'road_defect': request.form['road_defect'],
            'damage': request.form['damage'],
            'prim_contributory_cause': request.form['prim_contributory_cause'],
            'num_units': int(request.form['num_units']),
            'crash_hour': int(request.form['crash_hour']),
            'crash_month': int(request.form['crash_month']),
            'crash_dayofweek': int(request.form['crash_dayofweek'])
        }

        df_input = pd.DataFrame([user_input])

        # Encode
        for col in df_input.columns:
            if col in label_encoders:
                le = label_encoders[col]
                if df_input[col].iloc[0] not in le.classes_:
                    le.classes_ = np.append(le.classes_, df_input[col].iloc[0])
                df_input[col] = le.transform(df_input[col])

        # Match training columns
        X_train_columns = model.feature_names_in_
        for col in X_train_columns:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[X_train_columns]

        # Predict
        pred = model.predict(df_input)[0]
        if 'Accident_Severity' in label_encoders:
            prediction = label_encoders['Accident_Severity'].inverse_transform([pred])[0]
        else:
            prediction = pred

    return render_template('nmproject.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)