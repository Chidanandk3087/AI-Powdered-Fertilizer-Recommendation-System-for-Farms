import pandas as pd

# Load your dataset
df = pd.read_csv("fertilizer_recommendation_dataset.csv")

# Show all unique fertilizers
print("✅ Fertilizers found in dataset:")
print(df["Fertilizer"].unique())
