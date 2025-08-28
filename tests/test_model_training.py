import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_balanced_models import create_loan_dataset, train_balanced_model

def test_dataset_quality():
    df = create_loan_dataset()
    default_rate = df['default'].mean()
    
    assert len(df) > 1000, "Dataset should have sufficient samples"
    assert 0.1 <= default_rate <= 0.3, f"Default rate {default_rate:.3f} should be realistic"
    assert not df.isnull().any().any(), "Dataset should have no missing values"

def test_model_performance():
    # This will train a model and check it meets minimum standards
    success = train_balanced_model()
    assert success, "Model should pass validation with F1 >= 0.3"

if __name__ == "__main__":
    test_dataset_quality()
    test_model_performance()
    print("All tests passed!")
