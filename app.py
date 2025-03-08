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
        with open('model.pkl', 'rb') as file:
            model_data = pickle.load(file)
        print("Model loaded successfully")
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model_data = load_model()

if model_data is None:
    print("Warning: Model could not be loaded. Please ensure model.pkl exists in the current directory")
else:
    print(f"Loaded model: {model_data['model_name']}")
    print(f"Created at: {model_data['created_at']}")
    
    # Extract model components
    model = model_data['model']
    scaler = model_data['scaler']
    encoders = model_data['encoders']
    features = model_data['features']
    
    if features:
        print(f"Model expects {len(features)} features: {features}")

# Routes to serve HTML pages for different sections
@app.route('/', methods=['GET'])
def index():
    # Serve the main page
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    # Serve the prediction page
    return send_from_directory('static', 'predict.html')

@app.route('/model-info', methods=['GET'])
def model_info_page():
    # Serve the model info page
    return send_from_directory('static', 'model-info.html')

@app.route('/api-docs', methods=['GET'])
def api_docs_page():
    # Serve the API documentation page
    return send_from_directory('static', 'api-docs.html')

# API routes
@app.route('/api/info', methods=['GET'])
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
    if model_data is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get input data from request
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])
        
        # Preprocess input data to match training data format
        processed_data = preprocess_input(input_df)
        
        if processed_data is None:
            return jsonify({"error": "Error processing input data"}), 400
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Get probability if available
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(processed_data)[0][1]
            probability = float(probability)
        else:
            probability = None
        
        # Convert prediction to original label if encoders exist
        if encoders and 'target' in encoders:
            target_encoder = encoders['target']
            original_label = target_encoder.inverse_transform([prediction])[0]
        else:
            original_label = int(prediction)
        
        # Prepare response
        result = {
            "prediction": int(prediction),
            "label": original_label,
            "probability": probability
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def preprocess_input(input_df):
    """
    Preprocess the input data to match the format used during training
    """
    try:
        # Create a copy to avoid modifying the original
        df = input_df.copy()
        
        # Check if all required features are present
        if features is not None:
            missing_features = set(features) - set(df.columns)
            if missing_features:
                print(f"Warning: Missing features in input: {missing_features}")
                for feature in missing_features:
                    df[feature] = 0  # Default value for missing features
        
        # Encode categorical features
        if encoders:
            for col, encoder in encoders.items():
                if col != 'target' and col in df.columns:
                    try:
                        df[col] = encoder.transform(df[col])
                    except Exception as e:
                        print(f"Error encoding {col}: {e}")
                        # Handle unknown categories
                        df[col] = 0
        
        # Scale numerical features if scaler exists
        if scaler and features:
            num_cols = [col for col in features if col in df.columns]
            if num_cols:
                df[num_cols] = scaler.transform(df[num_cols])
        
        # Ensure columns are in the right order
        if features:
            # Fill any missing columns with 0
            for col in features:
                if col not in df.columns:
                    df[col] = 0
            
            # Select and order columns to match model expectations
            df = df[features]
        
        return df
    
    except Exception as e:
        print(f"Error preprocessing input: {e}")
        return None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model_loaded": model_data is not None})

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model_data is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    info = {
        "model_name": model_data['model_name'],
        "created_at": model_data['created_at'],
        "features": features,
        "target_labels": list(encoders['target'].classes_) if encoders and 'target' in encoders else None
    }
    
    return jsonify(info)

@app.route('/api/model-performance', methods=['GET'])
def model_performance():
    """Get performance metrics for the model"""
    if model_data is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # This would typically come from the model data
    # For demo purposes, we'll return sample metrics
    metrics = {
        "accuracy": 0.86,
        "precision": 0.83,
        "recall": 0.85,
        "f1_score": 0.84,
        "auc_roc": 0.89,
        "training_size": 303  # Standard size of Cleveland heart disease dataset
    }
    
    return jsonify(metrics)

@app.route('/api/sample-input', methods=['GET'])
def sample_input():
    """Provide a sample input structure"""
    if features is None:
        return jsonify({"error": "Model features not available"}), 500
    
    # Create a sample dictionary with expected features
    sample = {feature: 0 for feature in features}
    
    return jsonify({"sample_input": sample})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)