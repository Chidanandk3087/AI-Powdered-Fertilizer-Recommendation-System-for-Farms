import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import json

# Load dataset
dataset = pd.read_csv('fertilizer_recommendation_dataset.csv')

# Drop non-numeric, non-feature columns (like remarks)
if 'Remark' in dataset.columns:
    dataset = dataset.drop(['Remark'], axis=1)

if 'Fertilizer_Name' in dataset.columns:
    dataset = dataset.rename(columns={'Fertilizer_Name': 'Fertilizer'})

# Encode categorical columns
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

dataset['Soil'] = soil_encoder.fit_transform(dataset['Soil'])
dataset['Crop'] = crop_encoder.fit_transform(dataset['Crop'])
dataset['Fertilizer'] = fertilizer_encoder.fit_transform(dataset['Fertilizer'])

# Save encoders for app use
joblib.dump(soil_encoder, 'soil_encoder.pkl')
joblib.dump(crop_encoder, 'crop_encoder.pkl')
joblib.dump(fertilizer_encoder, 'fertilizer_encoder.pkl')

# Features and target
X = dataset.drop('Fertilizer', axis=1)
y = dataset['Fertilizer']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM': SVC(kernel='rbf', C=10, gamma=0.1, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=15, weights='distance', metric='manhattan'),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42)
}

model_accuracies = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train) * 100
    test_acc = model.score(X_test, y_test) * 100
    model_file = f"{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, model_file)
    print(f" {name} trained and saved as {model_file}")
    print(f" {name} Accuracy -> Train: {train_acc:.2f}%, Test: {test_acc:.2f}%")
    model_accuracies[name] = test_acc

# Save model accuracies
with open('model_accuracies.json', 'w') as f:
    json.dump(model_accuracies, f)

print(" All models and encoders saved successfully!")
