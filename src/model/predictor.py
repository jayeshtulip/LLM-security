# src/model/predictor.py - Main ML model predictor
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any, Optional
import logging
import os
from pathlib import Path

from src.api.schemas import LoanApplication, PredictionResponse

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing pipeline for loan applications"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame):
        """Fit preprocessing pipeline on training data"""
        # Encode categorical variables
        categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 
                              'Self_Employed', 'Property_Area']
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                # Handle missing values by adding 'Unknown' category
                df[col] = df[col].fillna('Unknown')
                le.fit(df[col])
                self.label_encoders[col] = le
        
        # Scale numerical features
        numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        numerical_data = df[numerical_columns].fillna(df[numerical_columns].median())
        self.scaler.fit(numerical_data)
        
        self.is_fitted = True
        logger.info("Data preprocessor fitted successfully")
        
    def transform(self, data):
        """Transform data using fitted preprocessing pipeline"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        if isinstance(data, LoanApplication):
            # Convert single LoanApplication to DataFrame
            df = pd.DataFrame([data.dict()])
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Data must be LoanApplication, dict, or DataFrame")
        
        # Handle categorical encoding
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                # Handle unseen categories
                df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else 'Unknown')
                df[col] = encoder.transform(df[col])
        
        # Handle numerical scaling
        numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        for col in numerical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
        
        numerical_data = df[numerical_columns]
        scaled_numerical = self.scaler.transform(numerical_data)
        
        # Combine all features
        feature_columns = list(self.label_encoders.keys()) + numerical_columns + ['Credit_History']
        
        # Ensure Credit_History is handled
        if 'Credit_History' in df.columns:
            df['Credit_History'] = df['Credit_History'].fillna(0)
        
        # Select and order features
        features = df[feature_columns]
        
        # Replace scaled numerical columns
        for i, col in enumerate(numerical_columns):
            features[col] = scaled_numerical[:, i]
            
        return features.values
    
    def fit_transform(self, df: pd.DataFrame):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)

class LoanPredictor:
    """Main loan prediction model"""
    
    def __init__(self, model_path: Optional[str] = None, preprocessor_path: Optional[str] = None):
        self.model = None
        self.preprocessor = None
        self.feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                             'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                             'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        
        # Try to load existing model and preprocessor
        if model_path and os.path.exists(model_path):
            self.load_model(model_path, preprocessor_path)
        else:
            # Train a new model if no existing model found
            logger.info("No existing model found, training new model...")
            self.train_model()
    
    def train_model(self):
        """Train a new loan prediction model with synthetic data"""
        logger.info("Training loan prediction model...")
        
        # Generate synthetic training data
        training_data = self._generate_synthetic_data(1000)
        
        # Prepare features and target
        X = training_data.drop('Loan_Status', axis=1)
        y = training_data['Loan_Status']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and fit preprocessor
        self.preprocessor = DataPreprocessor()
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        self.model.fit(X_train_processed, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"Model trained successfully - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Save model
        self.save_model()
        
    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic loan application data for training"""
        np.random.seed(42)
        
        data = {
            'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.8, 0.2]),
            'Married': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
            'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
            'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.8, 0.2]),
            'Self_Employed': np.random.choice(['No', 'Yes'], n_samples, p=[0.85, 0.15]),
            'ApplicantIncome': np.random.lognormal(8.5, 0.8, n_samples).astype(int),
            'CoapplicantIncome': np.random.lognormal(6.0, 1.2, n_samples).astype(int),
            'LoanAmount': np.random.normal(150, 50, n_samples).astype(int),
            'Loan_Amount_Term': np.random.choice([120, 180, 240, 300, 360], n_samples, 
                                               p=[0.05, 0.1, 0.1, 0.15, 0.6]),
            'Credit_History': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, 
                                            p=[0.4, 0.4, 0.2])
        }
        
        df = pd.DataFrame(data)
        
        # Generate target variable with realistic correlations
        approval_score = (
            (df['Credit_History'] * 0.4) +
            ((df['ApplicantIncome'] + df['CoapplicantIncome']) / 10000 * 0.3) +
            ((df['Education'] == 'Graduate').astype(int) * 0.15) +
            ((df['Property_Area'] == 'Urban').astype(int) * 0.1) +
            (np.random.random(n_samples) * 0.05)  # Random component
        )
        
        # Apply loan amount penalty
        loan_to_income_ratio = df['LoanAmount'] * 1000 / (df['ApplicantIncome'] + df['CoapplicantIncome'] + 1)
        approval_score -= np.clip(loan_to_income_ratio - 5, 0, 2) * 0.1
        
        df['Loan_Status'] = (approval_score > 0.5).astype(int)
        
        logger.info(f"Generated {n_samples} synthetic training samples")
        logger.info(f"Approval rate: {df['Loan_Status'].mean():.2%}")
        
        return df
    
    def predict(self, loan_application: LoanApplication) -> Dict[str, Any]:
        """Make loan prediction"""
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not loaded or trained")
        
        try:
            # Preprocess input
            X = self.preprocessor.transform(loan_application)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            # Get feature importance for explanation
            feature_importance = self.get_feature_importance()
            
            result = {
                "prediction": int(prediction),
                "prediction_label": "Approved" if prediction == 1 else "Rejected",
                "probability": {
                    "approved": float(probability[1]),
                    "rejected": float(probability[0])
                },
                "confidence": float(max(probability)),
                "risk_factors": self._analyze_risk_factors(loan_application, feature_importance),
                "model_version": "1.0.0"
            }
            
            logger.info(f"Prediction made: {result['prediction_label']} with {result['confidence']:.2%} confidence")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def _analyze_risk_factors(self, application: LoanApplication, feature_importance: Dict[str, float]) -> Dict[str, Any]:
        """Analyze risk factors for the application"""
        risk_factors = {}
        
        # Income analysis
        total_income = application.ApplicantIncome + application.CoapplicantIncome
        loan_amount_thousands = application.LoanAmount * 1000
        debt_to_income = loan_amount_thousands / max(total_income, 1)
        
        risk_factors["debt_to_income_ratio"] = round(debt_to_income, 2)
        risk_factors["income_category"] = "High" if total_income > 8000 else "Medium" if total_income > 4000 else "Low"
        
        # Credit history
        risk_factors["credit_history"] = "Good" if application.Credit_History == 1 else "Poor"
        
        # Key risk indicators
        high_risk_factors = []
        if debt_to_income > 6:
            high_risk_factors.append("High debt-to-income ratio")
        if application.Credit_History == 0:
            high_risk_factors.append("Poor credit history")
        if total_income < 3000:
            high_risk_factors.append("Low income")
            
        risk_factors["high_risk_factors"] = high_risk_factors
        risk_factors["risk_level"] = "High" if len(high_risk_factors) > 1 else "Medium" if len(high_risk_factors) == 1 else "Low"
        
        return risk_factors
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model"""
        if self.model is None:
            return {}
        
        importance_values = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance_values))
        
        # Sort by importance
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and information"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_type": type(self.model).__name__,
            "feature_count": len(self.feature_names),
            "features": self.feature_names,
            "feature_importance": self.get_feature_importance(),
            "model_version": "1.0.0",
            "training_date": "2024-01-01",  # Would be actual training date in production
            "status": "loaded"
        }
    
    def save_model(self, model_dir: str = "models"):
        """Save model and preprocessor"""
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "loan_model.joblib")
        preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    def load_model(self, model_path: str, preprocessor_path: Optional[str] = None):
        """Load saved model and preprocessor"""
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            if preprocessor_path and os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info(f"Preprocessor loaded from {preprocessor_path}")
            else:
                logger.warning("Preprocessor not found, will create new one if needed")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

# Convenience function for direct usage
def predict_loan_approval(loan_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to make loan predictions"""
    predictor = LoanPredictor()
    loan_application = LoanApplication(**loan_data)
    return predictor.predict(loan_application)