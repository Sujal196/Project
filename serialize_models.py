import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

print("Starting model training and serialization pipeline...")

# Load dataset
df = pd.read_csv('Dataset.csv')
print(f"Dataset loaded successfully. Shape: {df.shape}")

# Drop irrelevant columns for ML
cols_to_drop = ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2', 'Restaurant ID', 'Restaurant Name', 
                'Country Code', 'Address', 'Locality Verbose', 'Currency', 'Switch to order menu']
df_ml = df.drop(columns=[col for col in cols_to_drop if col in df.columns]).copy()
df_ml['Cuisines'] = df_ml['Cuisines'].fillna('Unknown')

# 1. Feature Engineering: Cuisines Multi-label encoding
cuisine_counts = df_ml['Cuisines'].str.split(',').explode().str.strip().value_counts()
top_30_cuisines = cuisine_counts.head(30).index.tolist()
all_cuisines = sorted(cuisine_counts.index.tolist())

print(f"Top 5 Cuisines: {top_30_cuisines[:5]}")

for cuisine in top_30_cuisines:
    df_ml[f'Cuisine_{cuisine}'] = df_ml['Cuisines'].apply(lambda x: 1 if cuisine in [c.strip() for c in x.split(',')] else 0)

# Encode binary categorical features
binary_cols = ['Has Table booking', 'Has Online delivery', 'Is delivering now']
for col in binary_cols:
    if col in df_ml.columns:
        df_ml[col] = df_ml[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

# 2. Supervised Regression Model (Gradient Boosting)
numeric_features = ['Average Cost for two', 'Price range', 'Votes', 'Has Table booking', 'Has Online delivery', 'Is delivering now']
cuisine_features = [f'Cuisine_{c}' for c in top_30_cuisines]
features = numeric_features + cuisine_features

X = df_ml[features]
y = df_ml['Aggregate rating']

# Train Scaler on full dataset
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])

# Train Champion Model
print("Training Gradient Boosting Regressor...")
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)
print("Gradient Boosting Regressor trained.")

# 3. Geospatial K-Means Clustering (New Delhi)
print("Performing K-Means Clustering on Delhi restaurants...")
delhi_df = df[(df['City'] == 'New Delhi') & (df['Longitude'] != 0) & (df['Latitude'] != 0)].copy()
coords = delhi_df[['Longitude', 'Latitude']]
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
delhi_df['Cluster'] = kmeans.fit_predict(coords)

# Keep only necessary fields for UI map plotting to save space
map_df = delhi_df[['Restaurant Name', 'Longitude', 'Latitude', 'Cluster', 'Aggregate rating', 'Average Cost for two', 'Cuisines']].copy()
print("K-Means clustering completed.")

# 4. Recommender Preprocessing (TF-IDF on Cuisines)
print("Preprocessing Recommender System data...")
rec_df = df[['Restaurant Name', 'Cuisines', 'Aggregate rating', 'Price range', 'City']].drop_duplicates(subset=['Restaurant Name']).copy()
rec_df['Cuisines'] = rec_df['Cuisines'].fillna('Unknown')
rec_df.reset_index(drop=True, inplace=True)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(rec_df['Cuisines'])
print("TF-IDF matrix built.")

# 5. Save metadata lists for Frontend
unique_restaurants = sorted(rec_df['Restaurant Name'].tolist())

# Save serialized models to joblib
print("Serializing and saving models...")
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump({
    'rec_df': rec_df,
    'tfidf_matrix': tfidf_matrix,
    'tfidf_vectorizer': tfidf
}, 'recommender.joblib')
joblib.dump({
    'map_df': map_df,
    'centroids': kmeans.cluster_centers_
}, 'clusters.joblib')
joblib.dump({
    'top_30_cuisines': top_30_cuisines,
    'all_cuisines': all_cuisines,
    'unique_restaurants': unique_restaurants
}, 'metadata.joblib')

print("All models serialized successfully! Verification files generated:")
print(" - model.joblib (Regressor)")
print(" - scaler.joblib (Scaler)")
print(" - recommender.joblib (TF-IDF matrix & metadata)")
print(" - clusters.joblib (Delhi restaurant coordinates & centroids)")
print(" - metadata.joblib (Autocomplete names and lists)")
