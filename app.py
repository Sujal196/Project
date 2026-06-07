from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Ensure models exist before starting
models_exist = all(os.path.exists(f) for f in ['model.joblib', 'scaler.joblib', 'recommender.joblib', 'clusters.joblib', 'metadata.joblib'])
if not models_exist:
    raise RuntimeError("Serialized model files not found! Please run 'python serialize_models.py' first.")

print("Loading serialized machine learning models...")
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
recommender_data = joblib.load('recommender.joblib')
clusters_data = joblib.load('clusters.joblib')
metadata = joblib.load('metadata.joblib')

# Extract assets
rec_df = recommender_data['rec_df']
tfidf_matrix = recommender_data['tfidf_matrix']
map_df = clusters_data['map_df']
top_30_cuisines = metadata['top_30_cuisines']
all_cuisines = metadata['all_cuisines']
unique_restaurants = metadata['unique_restaurants']

print("All ML models and metadata loaded successfully.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    return jsonify({
        'cuisines': all_cuisines,
        'top_30': top_30_cuisines,
        'restaurants': unique_restaurants,
        'success': True
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Extract fields
        average_cost = float(data.get('average_cost', 0))
        price_range = int(data.get('price_range', 1))
        votes = int(data.get('votes', 0))
        table_booking = int(data.get('table_booking', 0))
        online_delivery = int(data.get('online_delivery', 0))
        delivering_now = int(data.get('delivering_now', 0))
        selected_cuisines = data.get('cuisines', [])
        
        # Prepare numeric array (6 features)
        numeric_vals = np.array([[average_cost, price_range, votes, table_booking, online_delivery, delivering_now]])
        # Scale numeric features
        numeric_scaled = scaler.transform(numeric_vals)[0]
        
        # Prepare cuisine binary array (30 features)
        cuisine_vals = []
        for cuisine in top_30_cuisines:
            cuisine_vals.append(1 if cuisine in selected_cuisines else 0)
        
        # Combine numerical scaled features with binary cuisine features
        features_vector = np.concatenate([numeric_scaled, cuisine_vals]).reshape(1, -1)
        
        # Run prediction
        predicted_rating = float(model.predict(features_vector)[0])
        # Clip to valid range
        predicted_rating = max(0.0, min(5.0, round(predicted_rating, 2)))
        
        return jsonify({
            'rating': predicted_rating,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/recommend', methods=['GET'])
def recommend():
    try:
        name = request.args.get('name', '').strip()
        if not name:
            return jsonify({'error': 'Restaurant name is required'}), 400
        
        # Create case-insensitive search mapping
        indices = pd.Series(rec_df.index, index=rec_df['Restaurant Name'].str.lower()).drop_duplicates()
        name_lower = name.lower()
        
        if name_lower not in indices:
            # Try fuzzy matching (substring matching)
            matches = rec_df[rec_df['Restaurant Name'].str.lower().str.contains(name_lower)]
            if matches.empty:
                return jsonify({'error': f"Restaurant '{name}' not found."}), 404
            idx = matches.index[0]
        else:
            idx = indices[name_lower]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
        
        # Compute cosine similarity
        sim_array = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
        sim_scores = list(enumerate(sim_array))
        
        # Sort restaurants by similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top 5 excluding itself
        sim_scores = [item for item in sim_scores if item[0] != idx][:5]
        
        recommendations = []
        for index, score in sim_scores:
            row = rec_df.iloc[index]
            recommendations.append({
                'name': row['Restaurant Name'],
                'cuisines': row['Cuisines'],
                'rating': float(row['Aggregate rating']),
                'price_range': int(row['Price range']),
                'city': row['City'],
                'similarity': float(round(score, 4))
            })
            
        return jsonify({
            'query_restaurant': rec_df.iloc[idx]['Restaurant Name'],
            'recommendations': recommendations,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    try:
        # Convert map_df to JSON serializable list of dicts
        # Limit records to top 1500 to keep responses snappy (Map can cluster them)
        limited_df = map_df.sample(n=min(1500, len(map_df)), random_state=42)
        records = limited_df.to_dict(orient='records')
        return jsonify({
            'restaurants': records,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    print("Starting DineWise Web Application on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
