import os
import joblib
import numpy as np
import json
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from difflib import get_close_matches
import random

app = Flask(__name__)

# === Load Models and Encoders ===
base_path = os.path.dirname(__file__)

def load_model(filename):
    return joblib.load(os.path.join(base_path, filename))

model_rf = load_model('random_forest_model.pkl')
model_svm = load_model('svm_model.pkl')
model_knn = load_model('knn_model.pkl')
model_dt = load_model('decision_tree_model.pkl')

soil_encoder = load_model('soil_encoder.pkl')
crop_encoder = load_model('crop_encoder.pkl')
fertilizer_encoder = load_model('fertilizer_encoder.pkl')
scaler = load_model('scaler.pkl')

with open(os.path.join(base_path, 'model_accuracies.json'), 'r') as f:
    model_accuracies = json.load(f)

# === Fertilizer List from Dataset ===
dataset = pd.read_csv(os.path.join(base_path, 'fertilizer_recommendation_dataset.csv'))
fertilizer_list = dataset['Fertilizer'].unique()

# === Fertilizer Details for Display (FILENAMES now match your static folder) ===
fertilizer_display_data = {
    "Urea": {
        "name": "Urea",
        "remark": "Urea is one of the most commonly recommended nitrogen fertilizers because it provides a quick and concentrated source of nitrogen to crops. It helps in promoting leafy growth, improving photosynthesis, and increasing overall yield. Farmers prefer it since it is cost-effective, easy to store, and has a high nutrient content. For best results, it should be applied directly to the soil and incorporated with irrigation or tilling to minimize nitrogen loss.",
        "other_brands": ["Urea Gold", "NitroPlus"],
        "image": "urea.jpeg"
    },
    "DAP": {
        "name": "Di-Ammonium Phosphate (DAP)",
        "remark": "Di-Ammonium Phosphate (DAP) is recommended because it supplies both nitrogen and phosphorus, two essential nutrients for crop growth. It helps in strong root development, early plant establishment, and enhances flowering and fruiting. Farmers prefer it for its quick solubility and effectiveness in the initial stages of crop growth. It is usually applied at sowing time by mixing with soil to ensure better nutrient absorption.",
        "other_brands": ["DAP Supreme", "PhosRich"],
        "image": "dap.jpeg"
    },
    "Compost": {
        "name": "Compost",
        "remark": "Compost is recommended because it enriches the soil with organic matter and improves its overall fertility. It helps in retaining soil moisture, enhancing microbial activity, and providing a slow release of essential nutrients. Farmers and gardeners use compost to improve soil structure and reduce dependency on chemical fertilizers. It is applied by mixing into the soil or spreading as a top layer around crops for long-term soil health.",
        "other_brands": ["GreenGrow", "CompoCare"],
        "image": "compost.jpeg"
    },
    "Balanced NPK Fertilizer": {
        "name": "Balanced NPK Fertilizer",
        "remark": "Balanced NPK fertilizer is recommended because it provides all three major nutrients—Nitrogen (N), Phosphorus (P), and Potassium (K)—in equal proportions. It helps in promoting overall plant growth, from strong roots and healthy leaves to better flowering and fruit production. Farmers use it when crops need a uniform supply of nutrients at different stages of development. It is usually applied to the soil during planting or as a top dressing to maintain balanced nutrition.",
        "other_brands": ["GrowMore", "NPKMix"],
        "image": "npk.jpeg"
    },
    "Water Retaining Fertilizer": {
        "name": "Water Retaining Fertilizer",
        "remark": "Water retaining fertilizer is recommended because it improves the soil’s ability to hold water, reducing the need for frequent irrigation. It helps plants withstand dry conditions by ensuring a steady moisture supply to the roots. This type of fertilizer supports better nutrient uptake and reduces stress during drought. It is usually applied by mixing into the soil around crops, especially in water-scarce regions.",
        "other_brands": ["MoistureMax", "HydroHold"],
        "image": "water_retaining.jpeg"
    },
    "Organic Fertilizer": {
        "name": "Organic Fertilizer",
        "remark": "Organic fertilizer is recommended because it is made from natural sources like plant waste, animal manure, or bio-residues, making it safe and eco-friendly. It helps improve soil fertility, boosts microbial activity, and enhances long-term soil health. Unlike chemical fertilizers, it releases nutrients slowly, ensuring sustainable crop growth. Farmers apply it by mixing into the soil or using it as a top dressing to promote healthy and natural farming.",
        "other_brands": ["BioGrow", "EcoGreen"],
        "image": "orgainc-soil.jpeg"   # notice your file name has a typo "orgainc"
    },
    "Gypsum": {
        "name": "Gypsum",
        "remark": "Gypsum is recommended because it supplies calcium and sulfur, two important nutrients for crop growth. It helps in improving soil structure, reducing compaction, and enhancing root penetration. Gypsum also aids in reclaiming saline and sodic soils by flushing out excess sodium. Farmers usually apply it directly to the soil as a soil conditioner and nutrient source for better crop productivity.",
        "other_brands": ["SoilFix", "GypSoil"],
        "image": "gypsum.jpeg"
    },
    "Lime": {
        "name": "Lime",
        "remark": "Lime is recommended because it helps in reducing soil acidity and balancing soil pH, creating a better environment for crop growth. It improves nutrient availability, enhances microbial activity, and promotes stronger root development. Lime also supplies calcium and magnesium, which are essential for healthy plants. Farmers generally apply it by spreading and mixing into acidic soils before planting for long-term soil improvement.",
        "other_brands": ["Calcium Plus", "AgriLime"],
        "image": "lime.jpeg"
    },
    "General Purpose Fertilizer": {
        "name": "General Purpose Fertilizer",
        "remark": "General purpose fertilizer is recommended because it provides a balanced mix of essential nutrients suitable for a wide range of crops and soils. It helps in supporting overall plant health, encouraging steady growth, and improving yield quality. Farmers and gardeners use it when specific nutrient requirements are not known but general nourishment is needed. It is usually applied during planting or as a regular top dressing for consistent crop development.",
        "other_brands": ["AgroMix", "CropCare"],
        "image": "general_purpose.jpeg"
    },
    "Muriate of Potash": {
        "name": "Muriate of Potash",
        "remark": "Muriate of Potash (MOP) is recommended because it is a rich source of potassium, an essential nutrient for crop growth and disease resistance. It helps in improving water regulation, enhancing fruit quality, and strengthening plant stems. Farmers use it to boost flowering, fruiting, and overall yield, especially in crops like sugarcane, potatoes, and vegetables. It is usually applied directly to the soil or mixed with other fertilizers for balanced nutrition.",
        "other_brands": ["K+ Boost", "PotashPro"],
        "image": "potash.jpeg"
    }
}

# === Helper Functions ===
def safe_encode(value, encoder):
    classes = list(encoder.classes_)
    if value in classes:
        return encoder.transform([value])[0]
    matches = get_close_matches(value, classes, n=1, cutoff=0.6)
    if matches:
        return encoder.transform([matches[0]])[0]
    return encoder.transform([classes[0]])[0]  # fallback

def weighted_voting(predictions, weights):
    vote_scores = {}
    for pred, weight in zip(predictions, weights):
        vote_scores[pred] = vote_scores.get(pred, 0) + weight
    return max(vote_scores, key=vote_scores.get)

# === Routes ===
@app.route('/')
def home():
    return redirect(url_for('about'))

@app.route('/recommendation')
def recommendation():
    soil_types = list(soil_encoder.classes_)
    crop_types = list(crop_encoder.classes_)
    return render_template('index.html', soil_types=soil_types, crop_types=crop_types, result=None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_soil = request.form['Soil']
        user_crop = request.form['Crop']

        soil_code = safe_encode(user_soil, soil_encoder)
        crop_code = safe_encode(user_crop, crop_encoder)

        data = [
            float(request.form['Temperature']),
            float(request.form['Moisture']),
            float(request.form['Rainfall']),
            float(request.form['PH']),
            float(request.form['Nitrogen']),
            float(request.form['Phosphorous']),
            float(request.form['Potassium']),
            float(request.form['Carbon']),
            soil_code,
            crop_code
        ]

        features = scaler.transform([data])

        preds_encoded = [
            model_rf.predict(features)[0],
            model_svm.predict(features)[0],
            model_knn.predict(features)[0],
            model_dt.predict(features)[0]
        ]

        preds_labels = [fertilizer_encoder.inverse_transform([pred])[0] for pred in preds_encoded]

        weights = [
            model_accuracies.get('Random Forest', 1),
            model_accuracies.get('SVM', 1),
            model_accuracies.get('KNN', 1),
            model_accuracies.get('Decision Tree', 1)
        ]
        final_prediction = weighted_voting(preds_labels, weights)

        if final_prediction not in fertilizer_list:
            try:
                final_prediction = fertilizer_encoder.inverse_transform([preds_encoded[0]])[0]
            except:
                final_prediction = random.choice(fertilizer_list)

        recommended_fertilizer = fertilizer_display_data.get(final_prediction, {
            'name': final_prediction,
            'remark': "Details not available.",
            'other_brands': ["Regional brands may vary."],
            'image': "fert_generic.jpeg"
        })

        all_model_predictions = {
            "Random Forest": preds_labels[0],
            "SVM": preds_labels[1],
            "KNN": preds_labels[2],
            "Decision Tree": preds_labels[3],
            "Ensemble Result": final_prediction
        }

        return render_template('result.html',
                               result=recommended_fertilizer,
                               all_predictions=all_model_predictions,
                               message=None)

    except Exception as e:
        fallback_fertilizer = random.choice(fertilizer_list)
        recommended_fertilizer = {
            'name': fallback_fertilizer,
            'remark': "Details not available.",
            'other_brands': ["Regional brands may vary."],
            'image': "fert_generic.jpeg"
        }
        return render_template('result.html',
                               result=recommended_fertilizer,
                               all_predictions=None,
                               message=f"Error: {e}")

@app.route('/check_fertilizers')
def check_fertilizers():
    return render_template('fertilizers.html', fertilizers=fertilizer_display_data)

if __name__ == '__main__':
    app.run(debug=True)
