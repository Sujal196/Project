# DineWise: B2B Restaurant Analytics & Recommendation Portal

**DineWise** is a full-stack, data-driven restaurant intelligence portal designed to help entrepreneurs simulate success metrics, identify direct market competitors, and analyze geospatial dining clusters before opening a new location in New Delhi.

---

## 💡 What is DineWise?

Imagine you want to start a new food business, like an Italian cafe. Instead of spending millions of rupees and hoping it succeeds, **DineWise** acts as an AI-powered simulator to test your business concept beforehand:

1. **📊 Success Simulator (Predictive Modeling)**: 
   * Input your concept (cuisine, average cost, delivery options, target popularity).
   * DineWise runs a **Gradient Boosting Regressor** model trained on 9,500 real restaurants to predict your final customer rating.
   * Generates dynamic **AI Business Advice** (e.g., *"Premium restaurants without table reservations lose customer rating margin. Enable table bookings."*).

2. **🔍 Competitor Analysis (NLP Cuisine Matcher)**:
   * Type in any restaurant (e.g., "Subway").
   * The system uses **TF-IDF Text Analysis** and **Cosine Similarity** to scan all menus.
   * Displays the top 5 direct competitors selling matching food concepts in the city along with their ratings.

3. **🗺️ Market Mapping (Geospatial Clustering)**:
   * Displays an interactive dark-themed map of New Delhi.
   * Groups 1,500 active restaurants into **5 distinct K-Means clusters** based on price tiers and location.
   * Helps you inspect competitor pricing and pinpoint under-served streets to set up your business.

---

## 🛠️ Tech Stack & Machine Learning Details

* **Frontend**: HTML5, Vanilla CSS3 (SaaS Glassmorphic UI), Vanilla JavaScript, Leaflet.js (Map rendering via Canvas).
* **Backend**: Python, Flask (Microservice Architecture).
* **Machine Learning & NLP**: Scikit-Learn, Pandas, NumPy, Joblib.
  * **Rating Regression**: Gradient Boosting Regressor (Champion Model: **$R^2$ accuracy score = 95.8%**, Mean Absolute Error = 0.20 stars).
  * **Cuisine Matching**: TF-IDF Vectorization + Cosine Similarity.
  * **Locality Hubs**: K-Means Clustering ($k=5$).

---

## 📂 Project Structure

```text
├── Dataset.csv             # Raw Zomato restaurant dataset (9.5k+ records)
├── Clean.ipynb             # Initial data cleaning operations
├── Level1.ipynb            # Phase 1: Basic Descriptive Statistics & Cuisines EDA
├── Level2.ipynb            # Phase 2: Ratings, Chains, and Geospatial EDA
├── Level3_ML.ipynb         # Phase 3: Notebook demonstrating ML algorithms
├── serialize_models.py     # Script to train & export models to joblib
├── app.py                  # Flask web server (serves ML prediction endpoints)
├── templates/
│   └── index.html          # Interactive dark-mode dashboard UI
└── README.md               # Project documentation
```

---

## 🚀 How to Run the Web Application

### Prerequisite Libraries
Ensure you have the required Python libraries installed:
```bash
pip install pandas numpy scikit-learn flask joblib matplotlib seaborn
```

### Step 1: Serialize and Export the ML Models
Run the training script to pre-train and export the ML models:
```bash
python serialize_models.py
```
*This generates `model.joblib`, `scaler.joblib`, `recommender.joblib`, `clusters.joblib`, and `metadata.joblib` in your workspace.*

### Step 2: Start the Web Server
Launch the Flask backend server:
```bash
python app.py
```

### Step 3: Open the Portal
Open your web browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**
