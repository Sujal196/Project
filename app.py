from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pandas as pd
import numpy as np
import os
import sys

app = Flask(__name__, template_folder='templates', static_folder='static')

# ---------------------------------------------------------------
# Lazy Loading Pattern for Serverless Functions
# Models load on first request, not at cold start.
# This prevents timeout errors on Vercel.
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_CACHE = {}

def model_path(filename):
    return os.path.join(BASE_DIR, filename)

def load_models():
    """Lazy load all models on first request"""
    if MODELS_CACHE:
        return MODELS_CACHE
    
    print(f"BASE_DIR: {BASE_DIR}", file=sys.stderr)
    
    # Check if model files exist
    model_files = ['model.joblib', 'scaler.joblib', 'recommender.joblib', 'clusters.joblib', 'metadata.joblib']
    missing = [f for f in model_files if not os.path.exists(model_path(f))]
    if missing:
        error_msg = f"Model files not found: {missing}. Available files: {os.listdir(BASE_DIR)}"
        print(error_msg, file=sys.stderr)
        raise RuntimeError(error_msg)
    
    print("Loading models...", file=sys.stderr)
    try:
        MODELS_CACHE['model'] = joblib.load(model_path('model.joblib'))
        print("✓ model.joblib loaded", file=sys.stderr)
        
        MODELS_CACHE['scaler'] = joblib.load(model_path('scaler.joblib'))
        print("✓ scaler.joblib loaded", file=sys.stderr)
        
        recommender_data = joblib.load(model_path('recommender.joblib'))
        MODELS_CACHE['rec_df'] = recommender_data['rec_df']
        MODELS_CACHE['tfidf_matrix'] = recommender_data['tfidf_matrix']
        print("✓ recommender.joblib loaded", file=sys.stderr)
        
        clusters_data = joblib.load(model_path('clusters.joblib'))
        MODELS_CACHE['map_df'] = clusters_data['map_df']
        print("✓ clusters.joblib loaded", file=sys.stderr)
        
        metadata = joblib.load(model_path('metadata.joblib'))
        MODELS_CACHE['top_30_cuisines'] = metadata['top_30_cuisines']
        MODELS_CACHE['all_cuisines'] = metadata['all_cuisines']
        MODELS_CACHE['unique_restaurants'] = metadata['unique_restaurants']
        print("✓ metadata.joblib loaded", file=sys.stderr)
        
        print("All models loaded successfully!", file=sys.stderr)
        return MODELS_CACHE
    except Exception as e:
        import traceback
        error_msg = f"Model loading failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check - loads models on first call"""
    try:
        models = load_models()
        return jsonify({
            'status': 'healthy',
            'models_loaded': True,
            'message': 'All systems operational'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    try:
        models = load_models()
        return jsonify({
            'cuisines': models['all_cuisines'],
            'top_30': models['top_30_cuisines'],
            'restaurants': models['unique_restaurants'],
            'success': True
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        models = load_models()
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
        numeric_scaled = models['scaler'].transform(numeric_vals)[0]

        # Prepare cuisine binary array (30 features)
        cuisine_vals = [1 if cuisine in selected_cuisines else 0 for cuisine in models['top_30_cuisines']]

        # Combine & predict
        features_vector = np.concatenate([numeric_scaled, cuisine_vals]).reshape(1, -1)
        predicted_rating = float(models['model'].predict(features_vector)[0])
        predicted_rating = max(0.0, min(5.0, round(predicted_rating, 2)))

        return jsonify({'rating': predicted_rating, 'success': True})
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False
        }), 500

@app.route('/api/recommend', methods=['GET'])
def recommend():
    try:
        models = load_models()
        name = request.args.get('name', '').strip()
        if not name:
            return jsonify({'error': 'Restaurant name is required'}), 400

        # Case-insensitive index lookup
        rec_df = models['rec_df']
        indices = pd.Series(rec_df.index, index=rec_df['Restaurant Name'].str.lower()).drop_duplicates()
        name_lower = name.lower()

        if name_lower not in indices:
            matches = rec_df[rec_df['Restaurant Name'].str.lower().str.contains(name_lower, na=False)]
            if matches.empty:
                return jsonify({'error': f"Restaurant '{name}' not found."}), 404
            idx = matches.index[0]
        else:
            idx = indices[name_lower]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]

        # Cosine similarity
        tfidf_matrix = models['tfidf_matrix']
        sim_array = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
        sim_scores = sorted(enumerate(sim_array), key=lambda x: x[1], reverse=True)
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
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False
        }), 500

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    try:
        models = load_models()
        map_df = models['map_df']
        limited_df = map_df.sample(n=min(1500, len(map_df)), random_state=42)
        records = limited_df.to_dict(orient='records')
        return jsonify({'restaurants': records, 'success': True})
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False
        }), 500

# ---------------------------------------------------------------
# Local development entry point.
# Vercel does NOT call app.run() — it imports the `app` object
# directly as a WSGI callable. debug=False for production safety.
# ---------------------------------------------------------------
if __name__ == '__main__':
    print("Starting DineWise locally on http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
