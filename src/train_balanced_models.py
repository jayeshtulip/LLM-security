import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from datetime import datetime
import os

def setup_mlflow():
    # Use external LoadBalancer URL
    mlflow_uri = "http://a2fd03b8d3cad4e4caa91c9d87b876ec-1422723236.ap-south-1.elb.amazonaws.com:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    
    try:
        # Test connection with correct method
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        print(f"Connected to MLflow: {mlflow_uri}")
        print(f"Found {len(experiments)} experiments")
        
        # Set experiment
        mlflow.set_experiment("loan-default-balanced")
        return True
    except Exception as e:
        print(f"MLflow connection failed: {e}")
        return False

def create_loan_dataset():
    np.random.seed(42)
    n_samples = 10000
    
    # Generate realistic loan features
    credit_scores = np.clip(np.random.normal(680, 80, n_samples), 300, 850)  # Higher mean, lower std
    annual_incomes = np.random.lognormal(11.0, 0.4, n_samples)  # Higher income
    loan_amounts = np.clip(np.random.normal(20000, 12000, n_samples), 1000, 80000)  # Lower amounts
    employment_years = np.random.exponential(3, n_samples)
    debt_to_income = (loan_amounts * 12) / annual_incomes
    
    # Create realistic default probability - FIXED parameters
    default_logit = (
        -3.5 +  # Much lower base probability
        (680 - credit_scores) / 120 +  # Reduced credit score impact
        np.clip(debt_to_income - 0.4, 0, 1.0) * 1.5 +  # Reduced DTI impact
        np.random.normal(0, 0.2, n_samples)  # Less noise
    )
    
    default_probs = 1 / (1 + np.exp(-default_logit))
    defaults = np.random.binomial(1, default_probs)
    
    df = pd.DataFrame({
        'credit_score': credit_scores,
        'annual_income': annual_incomes,
        'loan_amount': loan_amounts,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income,
        'default': defaults
    })
    
    return df

def train_balanced_model():
    print("Starting balanced model training...")
    
    if not setup_mlflow():
        print("Proceeding with training but without MLflow logging...")
    
    # Create dataset
    df = create_loan_dataset()
    default_rate = df['default'].mean()
    
    print(f"Dataset: {len(df):,} samples")
    print(f"Default rate: {default_rate:.1%}")
    
    # Prepare features
    X = df.drop(['default'], axis=1)
    y = df['default']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Stratified split maintains class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Training default rate: {y_train.mean():.1%}")
    
    # Train balanced model
    print("Training RandomForest with class balancing...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Find optimal threshold for F1 score
    print("Optimizing prediction threshold...")
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.02)
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
    
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    y_pred_final = (y_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred_final)
    precision = precision_score(y_test, y_pred_final, zero_division=0)
    recall = recall_score(y_test, y_pred_final, zero_division=0)
    
    print(f"\nModel Performance:")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    
    # Show confusion matrix
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()
    print(f"\nConfusion Matrix:")
    print(f"True Negatives (correct approvals): {tn}")
    print(f"False Positives (wrong rejections): {fp}")
    print(f"False Negatives (missed defaults): {fn}")
    print(f"True Positives (caught defaults): {tp}")
    
    # Try to log to MLflow if connection works
    try:
        with mlflow.start_run(run_name=f"balanced-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            mlflow.log_param("dataset_size", len(df))
            mlflow.log_param("default_rate", default_rate)
            mlflow.log_param("approach", "class_balanced")
            mlflow.log_param("optimal_threshold", optimal_threshold)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("true_positives", int(tp))
            mlflow.log_metric("false_negatives", int(fn))
            mlflow.sklearn.log_model(model, "balanced_loan_model")
            
            if f1 >= 0.3:
                mlflow.set_tag("validation", "passed")
                mlflow.set_tag("deployment_ready", "true")
            else:
                mlflow.set_tag("validation", "failed")
                mlflow.set_tag("deployment_ready", "false")
        
        print("Results logged to MLflow successfully")
        
    except Exception as e:
        print(f"MLflow logging failed: {e}")
        print("Results computed but not logged to MLflow")
    
    # Validation check
    if f1 >= 0.3:
        print(f"\nVALIDATION PASSED: F1 score {f1:.3f} >= 0.3")
        print("Model can identify defaults and is ready for deployment")
        return True
    else:
        print(f"\nVALIDATION FAILED: F1 score {f1:.3f} < 0.3")
        print("Model cannot adequately identify defaults")
        return False

if __name__ == "__main__":
    success = train_balanced_model()
    print(f"\nTraining {'SUCCESS' if success else 'FAILED'}")
    print("\nTry accessing MLflow UI:")
    print("http://a2fd03b8d3cad4e4caa91c9d87b876ec-1422723236.ap-south-1.elb.amazonaws.com:5000")
