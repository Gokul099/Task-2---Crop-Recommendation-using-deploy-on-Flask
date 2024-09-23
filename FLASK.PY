import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the crop prediction model
model_filename = "crop_model.pkl"  
model = joblib.load(model_filename)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect features from the form
        Soil_color = request.form['Soil_color']
        Nitrogen = float(request.form['Nitrogen'])
        Phosphorus = float(request.form['Phosphorus'])
        Potassium = float(request.form['Potassium'])
        pH = float(request.form['pH'])
        Rainfall = float(request.form['Rainfall'])
        Temperature = float(request.form['Temperature'])

        # Map soil color to numerical values
        soil_color_mapping = {
            'red': 1,
            'black': 2,
            'alluvial': 3
        }
        Soil_color_numeric = soil_color_mapping.get(Soil_color.lower(), 0)

        # Prepare input features as a DataFrame
        input_features = pd.DataFrame({
            'Soil_color': [Soil_color_numeric],
            'Nitrogen': [Nitrogen],
            'Phosphorus': [Phosphorus],
            'Potassium': [Potassium],
            'pH': [pH],
            'Rainfall': [Rainfall],
            'Temperature': [Temperature]
        })

        # Make prediction
        prediction_numeric = model.predict(input_features)[0]  

        # Render the prediction result
        return render_template('pred.html', prediction=prediction_numeric)

if __name__ == '__main__':
    app.run(debug=True)
