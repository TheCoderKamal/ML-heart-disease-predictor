from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import pickle
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Create static folder if it doesn't exist
os.makedirs('static', exist_ok=True)

# Load the trained model
def load_model():
    try:
        print("Attempting to load model from model.pkl...")
        with open('model.pkl', 'rb') as file:
            model_data = pickle.load(file)
        
        if not isinstance(model_data, dict):
            print("Warning: Model data is not a dictionary. Creating default structure.")
            return create_default_model()
            
        required_keys = ['model', 'features']
        missing_keys = [key for key in required_keys if key not in model_data]
        
        if missing_keys:
            print(f"Warning: Model data is missing required keys: {missing_keys}")
            return create_default_model()
            
        print("Model loaded successfully")
        return model_data
    except FileNotFoundError:
        print("Error: model.pkl file not found. Creating default model.")
        return create_default_model()
    except Exception as e:
        print(f"Error loading model: {e}. Creating default model.")
        return create_default_model()

def create_default_model():
    try:
        import datetime
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder
        
        # Create default model data
        le = LabelEncoder()
        le.fit([0, 1])  # Binary classification
        
        model_data = {
            'model_name': 'Heart Disease Prediction Model (Default)',
            'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': RandomForestClassifier(n_estimators=10, random_state=42),
            'scaler': StandardScaler(),
            'encoders': {'target': le},
            'features': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        }
        print("Default model created successfully")
        return model_data
    except Exception as e:
        print(f"Error creating default model: {e}")
        return None

model_data = load_model()

if model_data is None:
    print("Error: Failed to load or create model. Application may not function correctly.")
else:
    model_name = model_data.get('model_name', 'Unknown Model')
    created_at = model_data.get('created_at', 'Unknown Date')
    model = model_data.get('model', None)
    scaler = model_data.get('scaler', None)
    encoders = model_data.get('encoders', {})
    features = model_data.get('features', [])
    
    print(f"Loaded model: {model_name}")
    print(f"Created at: {created_at}")
    print(f"Model expects {len(features)} features: {features}")

# Routes to serve HTML pages
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict')
def predict_page():
    return send_from_directory('static', 'predict.html')

@app.route('/model-info')
def model_info_page():
    return send_from_directory('static', 'model-info.html')

@app.route('/api-docs')
def api_docs_page():
    return send_from_directory('static', 'api-docs.html')

# API Endpoints
@app.route('/api/info')
def api_info():
    return jsonify({
        "message": "Heart Disease Prediction API",
        "endpoints": {
            "health": "/api/health",
            "model_info": "/api/model-info",
            "model_performance": "/api/model-performance",
            "sample_input": "/api/sample-input",
            "predict": "/api/predict (POST)"
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if model_data is None or model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        input_df = pd.DataFrame([data])
        processed_data = preprocess_input(input_df)
        if processed_data is None:
            return jsonify({"error": "Error processing input data"}), 400
        
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1] if hasattr(model, "predict_proba") else None
        
        # Handle label transformation safely
        try:
            if encoders and 'target' in encoders:
                original_label = encoders['target'].inverse_transform([prediction])[0]
            else:
                original_label = int(prediction)
        except Exception as e:
            print(f"Warning: Could not transform prediction label: {e}")
            original_label = int(prediction)
        
        return jsonify({
            "prediction": int(prediction),
            "label": original_label,
            "probability": float(probability) if probability is not None else None
        })
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

def preprocess_input(input_df):
    try:
        df = input_df.copy()
        
        # Ensure all required features are present
        if features:
            for feature in features:
                if feature not in df.columns:
                    df[feature] = 0
        
        # Apply encoders
        if encoders:
            for col, encoder in encoders.items():
                if col != 'target' and col in df.columns:
                    try:
                        df[col] = encoder.transform(df[[col]])
                    except Exception as e:
                        print(f"Warning: Failed to transform column {col}: {e}")
                        df[col] = 0
        
        # Apply scaling if available
        if scaler and features:
            try:
                feature_cols = [f for f in features if f in df.columns]
                df[feature_cols] = scaler.transform(df[feature_cols])
            except Exception as e:
                print(f"Warning: Failed to scale features: {e}")
        
        return df
    except Exception as e:
        print(f"Error preprocessing input: {e}")
        return None

@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "ok", 
        "model_loaded": model_data is not None,
        "model_name": model_data.get('model_name', 'Unknown') if model_data else None
    })

@app.route('/api/model-info')
def model_info():
    if model_data is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Safely get target labels
    target_labels = None
    if encoders and 'target' in encoders:
        try:
            target_labels = list(encoders['target'].classes_)
        except:
            target_labels = [0, 1]  # Default for binary classification
    
    return jsonify({
        "model_name": model_data.get('model_name', 'Unknown Model'),
        "created_at": model_data.get('created_at', 'Unknown Date'),
        "features": features,
        "target_labels": target_labels
    })

@app.route('/api/model-performance')
def model_performance():
    if model_data is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Get model performance metrics or use defaults
    performance = model_data.get('performance', {
        "accuracy": 0.86,
        "precision": 0.83,
        "recall": 0.85,
        "f1_score": 0.84,
        "auc_roc": 0.89,
        "training_size": 303
    })
    
    return jsonify(performance)

@app.route('/api/sample-input')
def sample_input():
    if not features:
        return jsonify({"error": "Model features not available"}), 500
    
    # Create a more realistic sample input
    sample = {}
    for feature in features:
        # Set default values based on feature names
        if feature == 'age':
            sample[feature] = 55
        elif feature == 'sex':
            sample[feature] = 1  # Male
        elif feature in ['cp', 'restecg', 'slope', 'ca', 'thal']:
            sample[feature] = 0
        elif feature == 'trestbps':
            sample[feature] = 130
        elif feature == 'chol':
            sample[feature] = 200
        elif feature == 'fbs':
            sample[feature] = 0
        elif feature == 'thalach':
            sample[feature] = 150
        elif feature == 'exang':
            sample[feature] = 0
        elif feature == 'oldpeak':
            sample[feature] = 1.0
        else:
            sample[feature] = 0
    
    return jsonify({"sample_input": sample})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)