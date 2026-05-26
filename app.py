from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class CropRecommender:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.encoder = None
        self.accuracies = {}
        self.load_models()

    def load_models(self):
        """Load all models and preprocessing objects"""
        model_files = {
            "Decision Tree": "crop_recommendation_dt.pkl",
            "Random Forest": "crop_recommendation_rf.pkl",
            "Logistic Regression": "crop_recommendation_lr.pkl",
            "SVM": "crop_recommendation_svm.pkl",
            "KNN": "crop_recommendation_knn.pkl"
        }

        try:
            data = pd.read_csv("Crop_recommendation.csv")
            X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
            y = data['label']
            
            self.encoder = LabelEncoder().fit(y)
            self.scaler = StandardScaler().fit(X)
            X_scaled = self.scaler.transform(X)
            y_encoded = self.encoder.transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42
            )

            for name, filename in model_files.items():
                try:
                    model = joblib.load(filename)
                    self.models[name] = model
                    self.accuracies[name] = round(accuracy_score(
                        y_test, model.predict(X_test)) * 100, 2)
                except Exception as e:
                    print(f"Error loading {name}: {str(e)}")
                    
            # Save preprocessing objects for future use
            joblib.dump(self.scaler, "scaler.pkl")
            joblib.dump(self.encoder, "encoder.pkl")
            
        except Exception as e:
            print(f"Loading fallback: {str(e)}")
            self.load_fallback_models(model_files)

    def load_fallback_models(self, model_files):
        """Fallback loading without dataset"""
        for name, filename in model_files.items():
            try:
                self.models[name] = joblib.load(filename)
                self.accuracies[name] = 0.0  # Default accuracy when we can't calculate
            except Exception as e:
                print(f"Error loading {name}: {str(e)}")
        
        try:
            self.scaler = joblib.load("scaler.pkl")
            self.encoder = joblib.load("encoder.pkl")
        except Exception as e:
            print(f"Error loading preprocessing: {str(e)}")

    def predict(self, data, algorithm="Random Forest"):
        """Make prediction with error handling"""
        if not self.models or not self.scaler or not self.encoder:
            return None, "System not properly initialized"
        
        if algorithm not in self.models:
            return None, f"Algorithm {algorithm} not available"
        
        try:
            input_data = np.array([[
                float(data['N']), float(data['P']), float(data['K']),
                float(data['temperature']), float(data['humidity']),
                float(data['ph']), float(data['rainfall'])
            ]])
            
            scaled_input = self.scaler.transform(input_data)
            prediction = self.models[algorithm].predict(scaled_input)
            return self.encoder.inverse_transform(prediction)[0], None
            
        except Exception as e:
            return None, str(e)

# Initialize recommender system
recommender = CropRecommender()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        algorithm = data.get('algorithm', 'Random Forest')
        
        # Validate required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        crop, error = recommender.predict(data, algorithm)
        if error:
            return jsonify({
                "error": error,
                "available_algorithms": list(recommender.models.keys())
            }), 400
        
        return jsonify({
            "crop": crop,
            "algorithm": algorithm,
            "accuracy": recommender.accuracies.get(algorithm, "N/A")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Validate required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        results = {}
        for algo in recommender.models:
            crop, error = recommender.predict(data, algo)
            results[algo] = {
                "crop": crop if not error else None,
                "error": error,
                "accuracy": recommender.accuracies.get(algo, "N/A")
            }
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\nSystem Initialization Report:")
    print(f"Loaded {len(recommender.models)} models")
    print("Available algorithms:", list(recommender.models.keys()))
    
    if not recommender.scaler or not recommender.encoder:
        print("Warning: Preprocessing tools not fully loaded")
    
    app.run(host='0.0.0.0', port=5000, debug=True)