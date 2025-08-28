# src/api/schemas.py - Pydantic schemas for API requests and responses
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
from datetime import datetime

class LoanApplication(BaseModel):
    """Loan application input schema"""
    Gender: str = Field(..., description="Applicant gender", example="Male")
    Married: str = Field(..., description="Marital status", example="Yes") 
    Dependents: str = Field(..., description="Number of dependents", example="1")
    Education: str = Field(..., description="Education level", example="Graduate")
    Self_Employed: str = Field(..., description="Self employment status", example="No")
    ApplicantIncome: int = Field(..., description="Applicant income in currency units", example=5849, gt=0)
    CoapplicantIncome: int = Field(..., description="Co-applicant income in currency units", example=0, ge=0)
    LoanAmount: int = Field(..., description="Loan amount in thousands", example=120, gt=0)
    Loan_Amount_Term: int = Field(..., description="Loan term in months", example=360, gt=0)
    Credit_History: int = Field(..., description="Credit history (1=good, 0=poor)", example=1)
    Property_Area: str = Field(..., description="Property area type", example="Urban")
    
    @validator('Gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be Male or Female')
        return v
    
    @validator('Married')
    def validate_married(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('Married must be Yes or No')
        return v
    
    @validator('Dependents')
    def validate_dependents(cls, v):
        if v not in ['0', '1', '2', '3+']:
            raise ValueError('Dependents must be 0, 1, 2, or 3+')
        return v
    
    @validator('Education')
    def validate_education(cls, v):
        if v not in ['Graduate', 'Not Graduate']:
            raise ValueError('Education must be Graduate or Not Graduate')
        return v
    
    @validator('Self_Employed')
    def validate_self_employed(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('Self_Employed must be Yes or No')
        return v
    
    @validator('Credit_History')
    def validate_credit_history(cls, v):
        if v not in [0, 1]:
            raise ValueError('Credit_History must be 0 or 1')
        return v
    
    @validator('Property_Area')
    def validate_property_area(cls, v):
        if v not in ['Urban', 'Semiurban', 'Rural']:
            raise ValueError('Property_Area must be Urban, Semiurban, or Rural')
        return v
    
    @validator('Loan_Amount_Term')
    def validate_loan_term(cls, v):
        valid_terms = [120, 180, 240, 300, 360]
        if v not in valid_terms:
            raise ValueError(f'Loan_Amount_Term must be one of {valid_terms}')
        return v

class RiskFactors(BaseModel):
    """Risk factors analysis schema"""
    debt_to_income_ratio: float = Field(..., description="Debt to income ratio")
    income_category: str = Field(..., description="Income category (Low/Medium/High)")
    credit_history: str = Field(..., description="Credit history status")
    high_risk_factors: List[str] = Field(..., description="List of high risk factors")
    risk_level: str = Field(..., description="Overall risk level (Low/Medium/High)")

class PredictionResponse(BaseModel):
    """Loan prediction response schema"""
    prediction: int = Field(..., description="Prediction result (0=Rejected, 1=Approved)")
    prediction_label: str = Field(..., description="Human readable prediction")
    probability: Dict[str, float] = Field(..., description="Prediction probabilities")
    confidence: float = Field(..., description="Model confidence (0-1)")
    risk_factors: RiskFactors = Field(..., description="Risk analysis")
    model_version: str = Field(..., description="Model version used")
    request_id: Optional[str] = Field(None, description="Request identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    timestamp: Optional[datetime] = Field(None, description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Service status", example="healthy")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version", example="1.0.0")
    model_status: str = Field(..., description="Model loading status")

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ModelInfo(BaseModel):
    """Model information schema"""
    model_type: str = Field(..., description="Type of ML model")
    feature_count: int = Field(..., description="Number of features")
    features: List[str] = Field(..., description="Feature names")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    model_version: str = Field(..., description="Model version")
    training_date: str = Field(..., description="Model training date")
    status: str = Field(..., description="Model status")

class AuthToken(BaseModel):
    """Authentication token schema"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(..., description="Token type (bearer)")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    permissions: List[str] = Field(..., description="User permissions")

class ValidationResult(BaseModel):
    """Model validation result schema"""
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    validation_results: Dict[str, bool] = Field(..., description="Validation gate results")
    overall_passed: bool = Field(..., description="Overall validation status")
    gates: Dict[str, float] = Field(..., description="Validation thresholds")
    test_samples: int = Field(..., description="Number of test samples used")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema"""
    applications: List[LoanApplication] = Field(..., description="List of loan applications")
    return_probabilities: bool = Field(True, description="Include prediction probabilities")
    return_risk_analysis: bool = Field(True, description="Include risk factor analysis")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema"""
    predictions: List[PredictionResponse] = Field(..., description="Prediction results")
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")
    processing_time: float = Field(..., description="Total processing time in seconds")

# Example data for documentation
class Examples:
    """Example data for API documentation"""
    
    LOAN_APPLICATION_EXAMPLE = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate", 
        "Self_Employed": "No",
        "ApplicantIncome": 5849,
        "CoapplicantIncome": 0,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban"
    }
    
    PREDICTION_RESPONSE_EXAMPLE = {
        "prediction": 1,
        "prediction_label": "Approved",
        "probability": {
            "approved": 0.78,
            "rejected": 0.22
        },
        "confidence": 0.78,
        "risk_factors": {
            "debt_to_income_ratio": 2.46,
            "income_category": "Medium",
            "credit_history": "Good",
            "high_risk_factors": [],
            "risk_level": "Low"
        },
        "model_version": "1.0.0"
    }