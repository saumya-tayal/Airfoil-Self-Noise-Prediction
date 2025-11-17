import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import os

class AirfoilModel:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_names = ['Frequency', 'Angle', 'ChordLength', 'Velocity', 'Thickness']
        self.performance_metrics = {}
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.full_dataset = None
        
    def train_model(self, model_type='rf', n_estimators=100, max_depth=10):
        """Train the model with parameters from frontend"""
        try:
            # Load data
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
            df = pd.read_csv(url, sep=r'\s+', header=None,
                           names=['Frequency', 'Angle', 'ChordLength', 'Velocity', 'Thickness', 'SoundPressure'])
            
            # Store full dataset for visualization
            self.full_dataset = df
            
            # Clean data
            df_clean = df.drop_duplicates()
            
            # Prepare features and target
            X = df_clean[self.feature_names]
            y = df_clean['SoundPressure']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            if model_type == 'rf':
                # Use parameters from frontend
                param_grid = {
                    'n_estimators': [n_estimators],
                    'max_depth': [max_depth],
                    'min_samples_split': [2, 5]
                }
                
                grid_search = GridSearchCV(
                    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
                    param_grid=param_grid,
                    scoring='r2',
                    cv=3,
                    n_jobs=-1,
                    verbose=0
                )
                
                print("Training Random Forest model...")
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                
            elif model_type == 'lr':
                print("Training Linear Regression model...")
                self.model = LinearRegression()
                self.model.fit(X_train, y_train)
            
            self.is_trained = True
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            # Store data for visualization
            self.X_test = X_test
            self.y_test = y_test
            self.y_pred = y_pred
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation scores
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            
            # Store performance metrics
            self.performance_metrics = {
                'mae': round(float(mae), 3),
                'r2_score': round(float(r2), 3),
                'cv_score': round(float(cv_mean), 3),
                'best_params': getattr(grid_search, 'best_params_', {'model': 'LinearRegression'}) if model_type == 'rf' else {'model': 'LinearRegression'},
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'model_type': model_type
            }
            
            # Save model
            joblib.dump(self.model, 'trained_model.pkl')
            
            return {
                'status': 'success',
                'message': f'{model_type.upper()} model trained successfully',
                'metrics': self.performance_metrics
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            if os.path.exists('trained_model.pkl'):
                self.model = joblib.load('trained_model.pkl')
                self.is_trained = True
                return {'status': 'success', 'message': 'Model loaded successfully'}
            else:
                return {'status': 'error', 'message': 'No trained model found. Please train first.'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, input_data):
        """Make predictions from frontend input"""
        if not self.is_trained:
            return {'status': 'error', 'message': 'Model not trained. Please train first.'}
        
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data], columns=self.feature_names)
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            
            return {
                'status': 'success',
                'prediction': round(float(prediction), 2),
                'units': 'dB',
                'input_features': input_data
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Prediction error: {str(e)}'}
    
    def get_model_info(self):
        """Get model information"""
        if not self.is_trained:
            return {'status': 'error', 'message': 'Model not trained'}
        
        return {
            'status': 'success',
            'model_type': type(self.model).__name__,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'target_name': 'SoundPressure'
        }
    
    def get_metrics(self):  # FIXED: Changed from get_performance_metrics to get_metrics
        """Get model performance metrics for frontend display"""
        if not self.is_trained:
            return {'status': 'error', 'message': 'Model not trained'}
        
        return {
            'status': 'success',
            'metrics': self.performance_metrics
        }
    
    def get_feature_importance(self):
        """Get feature importance for frontend display"""
        if not self.is_trained:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_imp = dict(zip(self.feature_names, importances))
                
                # Sort by importance
                sorted_importance = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True))
                
                return {
                    'status': 'success',
                    'feature_importance': sorted_importance
                }
            else:
                return {
                    'status': 'success',
                    'feature_importance': {feature: 0.2 for feature in self.feature_names},
                    'message': 'Feature importance not available for this model type'
                }
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_data_for_visualization(self):
        """Get test data and predictions for frontend visualization"""
        if not self.is_trained or self.full_dataset is None:
            # Return empty data structure if no model trained
            return {
                'features': {col: [] for col in self.feature_names},
                'target': [],
                'y_test': [],
                'y_pred': [],
                'feature_names': self.feature_names
            }
        
        # Return actual trained data
        return {
            'features': {col: self.full_dataset[col].tolist() for col in self.feature_names},
            'target': self.full_dataset['SoundPressure'].tolist(),
            'y_test': self.y_test.tolist() if self.y_test is not None else [],
            'y_pred': self.y_pred.tolist() if self.y_pred is not None else [],
            'feature_names': self.feature_names
        }

# Global model instance
airfoil_model = AirfoilModel()