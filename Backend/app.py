from flask import Flask, request, jsonify
from flask_cors import CORS
from model import airfoil_model
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return jsonify({
        'message': 'Airfoil Sound Pressure Prediction API',
        'status': 'running',
        'endpoints': {
            '/train': 'POST - Train the model',
            '/predict': 'POST - Make predictions',
            '/model-info': 'GET - Get model information',
            '/metrics': 'GET - Get model performance metrics',
            '/feature-importance': 'GET - Get feature importance',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_trained': airfoil_model.is_trained,
        'model_ready': airfoil_model.is_trained
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the current model"""
    info = airfoil_model.get_model_info()
    return jsonify(info)

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get model performance metrics"""
    metrics = airfoil_model.get_metrics()
    return jsonify(metrics)

@app.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance data"""
    importance = airfoil_model.get_feature_importance()
    return jsonify(importance)

@app.route('/data', methods=['GET'])
def get_data():
    """Get training data and predictions for visualization"""
    try:
        data_dict = airfoil_model.get_data_for_visualization()
        return jsonify({
            'status': 'success',
            'data': data_dict
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model with parameters from frontend"""
    try:
        data = request.get_json() or {}
        
        # Get parameters from frontend or use defaults
        model_type = data.get('model_type', 'rf')
        n_estimators = data.get('n_estimators', 100)
        max_depth = data.get('max_depth', 10)
        
        result = airfoil_model.train_model(
            model_type=model_type,
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions from frontend input"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        # Extract features from frontend input
        features = {
            'Frequency': data.get('frequency'),
            'Angle': data.get('angle'),
            'ChordLength': data.get('chordLength'),
            'Velocity': data.get('velocity'),
            'Thickness': data.get('thickness')
        }
        
        # Validate that all features are provided
        missing_features = [key for key, value in features.items() if value is None]
        if missing_features:
            return jsonify({
                'status': 'error', 
                'message': f'Missing features: {missing_features}'
            }), 400
        
        result = airfoil_model.predict(features)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/load-model', methods=['POST'])
def load_model():
    """Load a pre-trained model"""
    result = airfoil_model.load_model()
    return jsonify(result)

if __name__ == '__main__':
    # Try to load existing model on startup
    airfoil_model.load_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)