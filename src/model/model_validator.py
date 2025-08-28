# src/model/model_validator.py - Model performance validation for CI/CD gates
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Tuple
import joblib
import logging

logger = logging.getLogger(__name__)

class ModelValidationError(Exception):
    """Custom exception for model validation failures"""
    pass

class ModelPerformanceValidator:
    """Validates model performance against quality gates"""
    
    def __init__(self, min_f1_score: float = 0.3, min_accuracy: float = 0.6):
        self.min_f1_score = min_f1_score
        self.min_accuracy = min_accuracy
        
    def validate_model_performance(self, model, preprocessor, X_test=None, y_test=None) -> Dict[str, Any]:
        """
        Validate model performance against quality gates
        
        Args:
            model: Trained model
            preprocessor: Data preprocessor
            X_test: Test features (optional - will load default test set if not provided)
            y_test: Test labels (optional - will load default test set if not provided)
            
        Returns:
            Dict with performance metrics and validation results
        """
        
        try:
            # Load test data if not provided
            if X_test is None or y_test is None:
                X_test, y_test = self._load_test_data()
            
            # Preprocess test data
            if preprocessor is not None:
                X_test_processed = preprocessor.transform(X_test)
            else:
                X_test_processed = X_test
            
            # Make predictions
            y_pred = model.predict(X_test_processed)
            y_pred_proba = None
            
            try:
                y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            except AttributeError:
                logger.warning("Model doesn't support predict_proba, skipping AUC calculation")
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            if y_pred_proba is not None:
                try:
                    metrics['auc_score'] = roc_auc_score(y_test, y_pred_proba)
                except ValueError as e:
                    logger.warning(f"Could not calculate AUC score: {e}")
                    metrics['auc_score'] = None
            
            # Cross-validation metrics for robustness
            cv_scores = cross_val_score(model, X_test_processed, y_test, cv=3, scoring='f1_weighted')
            metrics['cv_f1_mean'] = cv_scores.mean()
            metrics['cv_f1_std'] = cv_scores.std()
            
            # Validation results
            validation_results = {
                'f1_gate_passed': metrics['f1_score'] >= self.min_f1_score,
                'accuracy_gate_passed': metrics['accuracy'] >= self.min_accuracy,
                'cv_stability_good': metrics['cv_f1_std'] < 0.1  # Low standard deviation indicates stability
            }
            
            # Overall validation status
            overall_passed = (
                validation_results['f1_gate_passed'] and 
                validation_results['accuracy_gate_passed']
            )
            
            result = {
                'metrics': metrics,
                'validation_results': validation_results,
                'overall_passed': overall_passed,
                'gates': {
                    'min_f1_score': self.min_f1_score,
                    'min_accuracy': self.min_accuracy
                },
                'test_samples': len(y_test)
            }
            
            # Log results
            self._log_validation_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise ModelValidationError(f"Model validation failed: {e}")
    
    def _load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load default test dataset"""
        try:
            # Try to load saved test data
            X_test = pd.read_csv('data/X_test.csv')
            y_test = pd.read_csv('data/y_test.csv').squeeze()
            return X_test, y_test
        except FileNotFoundError:
            logger.warning("Test data files not found, generating synthetic test data")
            return self._generate_synthetic_test_data()
    
    def _generate_synthetic_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic test data for validation (fallback)"""
        np.random.seed(42)
        
        # Generate synthetic loan application data
        n_samples = 100
        
        data = {
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Married': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
            'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
            'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
            'ApplicantIncome': np.random.randint(1000, 15000, n_samples),
            'CoapplicantIncome': np.random.randint(0, 8000, n_samples),
            'LoanAmount': np.random.randint(50, 500, n_samples),
            'Loan_Amount_Term': np.random.choice([120, 180, 240, 300, 360], n_samples),
            'Credit_History': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples)
        }
        
        X_test = pd.DataFrame(data)
        
        # Generate synthetic labels with some correlation to features
        approval_probability = (
            (X_test['Credit_History'] * 0.4) +
            (X_test['ApplicantIncome'] / 10000 * 0.3) +
            ((X_test['Education'] == 'Graduate').astype(int) * 0.2) +
            (np.random.random(n_samples) * 0.1)
        )
        
        y_test = (approval_probability > 0.5).astype(int)
        
        logger.info(f"Generated {n_samples} synthetic test samples")
        return X_test, pd.Series(y_test)
    
    def _log_validation_results(self, result: Dict[str, Any]):
        """Log validation results"""
        metrics = result['metrics']
        validation = result['validation_results']
        
        logger.info("="*50)
        logger.info("MODEL VALIDATION RESULTS")
        logger.info("="*50)
        logger.info(f"Test samples: {result['test_samples']}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        
        if metrics.get('auc_score'):
            logger.info(f"AUC Score: {metrics['auc_score']:.4f}")
        
        logger.info(f"CV F1 Mean: {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
        
        logger.info("\nGATE RESULTS:")
        logger.info(f"F1 Score Gate (≥ {result['gates']['min_f1_score']}): {'✅ PASS' if validation['f1_gate_passed'] else '❌ FAIL'}")
        logger.info(f"Accuracy Gate (≥ {result['gates']['min_accuracy']}): {'✅ PASS' if validation['accuracy_gate_passed'] else '❌ FAIL'}")
        logger.info(f"CV Stability: {'✅ STABLE' if validation['cv_stability_good'] else '⚠️ UNSTABLE'}")
        
        logger.info(f"\nOVERALL: {'✅ VALIDATION PASSED' if result['overall_passed'] else '❌ VALIDATION FAILED'}")
        logger.info("="*50)


def validate_model_performance(model, preprocessor, min_f1_score: float = 0.3) -> Dict[str, Any]:
    """
    Convenience function for CI/CD pipeline model validation
    
    Args:
        model: Trained model
        preprocessor: Data preprocessor
        min_f1_score: Minimum F1 score required to pass validation
        
    Returns:
        Dict with validation results
        
    Raises:
        ModelValidationError: If validation fails
    """
    validator = ModelPerformanceValidator(min_f1_score=min_f1_score)
    result = validator.validate_model_performance(model, preprocessor)
    
    if not result['overall_passed']:
        failed_gates = []
        if not result['validation_results']['f1_gate_passed']:
            failed_gates.append(f"F1 score ({result['metrics']['f1_score']:.4f}) < {min_f1_score}")
        if not result['validation_results']['accuracy_gate_passed']:
            failed_gates.append(f"Accuracy ({result['metrics']['accuracy']:.4f}) < {validator.min_accuracy}")
        
        raise ModelValidationError(f"Model validation failed: {', '.join(failed_gates)}")
    
    return result

# CLI for running validation manually
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate model performance')
    parser.add_argument('--model-path', required=True, help='Path to trained model file')
    parser.add_argument('--preprocessor-path', help='Path to preprocessor file')
    parser.add_argument('--min-f1', type=float, default=0.3, help='Minimum F1 score threshold')
    parser.add_argument('--min-accuracy', type=float, default=0.6, help='Minimum accuracy threshold')
    
    args = parser.parse_args()
    
    try:
        # Load model
        model = joblib.load(args.model_path)
        
        # Load preprocessor if provided
        preprocessor = None
        if args.preprocessor_path:
            preprocessor = joblib.load(args.preprocessor_path)
        
        # Run validation
        validator = ModelPerformanceValidator(
            min_f1_score=args.min_f1,
            min_accuracy=args.min_accuracy
        )
        
        result = validator.validate_model_performance(model, preprocessor)
        
        # Exit with appropriate code
        sys.exit(0 if result['overall_passed'] else 1)
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)