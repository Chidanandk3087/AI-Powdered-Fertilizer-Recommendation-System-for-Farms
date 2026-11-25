import joblib

# Load encoders from current directory
soil_encoder = joblib.load('soil_encoder.pkl')
crop_encoder = joblib.load('crop_encoder.pkl')

# Print the classes
print(" Soil types:", soil_encoder.classes_)
print(" Crop types:", crop_encoder.classes_)
