# src/main.py - Enhanced API with GET endpoints for browser testing
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Import your model components
from src.model.predictor import LoanPredictor
from src.api.schemas import LoanApplication, PredictionResponse, HealthResponse

# Import the working LLM component
try:
    from src.llm.simple_analyzer import generate_enhanced_explanation
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  LLM not available - using basic predictions only")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Enhanced schemas for LLM responses
class LLMExplanation(BaseModel):
    """LLM explanation schema"""
    explanation: str = Field(..., description="AI-generated explanation of the decision")
    advice: List[str] = Field(..., description="Personalized advice for the applicant")
    llm_available: bool = Field(..., description="Whether LLM was available for generation")
    generated_at: str = Field(..., description="When the explanation was generated")
    model_type: Optional[str] = Field(None, description="Type of model used for explanation")

class EnhancedPredictionResponse(BaseModel):
    """Enhanced prediction response with LLM explanations"""
    # Original ML prediction fields
    prediction: int = Field(..., description="Prediction result (0=Rejected, 1=Approved)")
    prediction_label: str = Field(..., description="Human readable prediction")
    probability: Dict[str, float] = Field(..., description="Prediction probabilities")
    confidence: float = Field(..., description="Model confidence (0-1)")
    risk_factors: Dict[str, Any] = Field(..., description="Risk analysis")
    model_version: str = Field(..., description="Model version used")
    
    # Enhanced LLM fields
    ai_explanation: LLMExplanation = Field(..., description="AI-generated explanation")
    
    # Metadata
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: Optional[datetime] = Field(None, description="Prediction timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Total processing time")

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Loan Prediction API with AI Explanations",
    description="ML API for loan approval prediction with intelligent explanations",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global predictor
    try:
        logger.info("Initializing ML predictor...")
        predictor = LoanPredictor()
        logger.info("✅ ML model loaded successfully!")
        
        if LLM_AVAILABLE:
            logger.info("✅ LLM explanation system ready!")
        else:
            logger.warning("⚠️ LLM not available - using basic explanations")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Loan Prediction API with AI Explanations",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health - Service health check",
            "docs": "/docs - Interactive API documentation", 
            "basic_example": "/example - Test basic prediction (GET)",
            "enhanced_example": "/example/enhanced - Test enhanced prediction (GET)",
            "predict": "/predict - Basic prediction (POST)",
            "predict_enhanced": "/predict/enhanced - Enhanced prediction with AI (POST)"
        },
        "ml_available": predictor is not None,
        "ai_explanations": LLM_AVAILABLE
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check including LLM status"""
    try:
        ml_status = "healthy" if predictor and predictor.model is not None else "unhealthy"
        llm_status = "available" if LLM_AVAILABLE else "unavailable"
        
        overall_status = "healthy" if ml_status == "healthy" else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="2.0.0",
            model_status=f"ML: {ml_status}, LLM: {llm_status}"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.get("/ready")
async def readiness_check():
    """Enhanced readiness check"""
    try:
        if predictor is None or predictor.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model not ready"
            )
        
        return {
            "status": "ready",
            "ml_model": "available",
            "explanation_system": "available" if LLM_AVAILABLE else "basic_mode"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )

# GET endpoints for browser testing
@app.get("/example")
async def get_basic_example():
    """Test basic prediction with example data (GET endpoint for browser)"""
    try:
        example_data = LoanApplication(
            Gender="Male",
            Married="Yes", 
            Dependents="1",
            Education="Graduate",
            Self_Employed="No",
            ApplicantIncome=5849,
            CoapplicantIncome=0,
            LoanAmount=120,
            Loan_Amount_Term=360,
            Credit_History=1,
            Property_Area="Urban"
        )
        
        return await predict_loan_basic(example_data)
    except Exception as e:
        logger.error(f"Basic example error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/example/enhanced")
async def get_enhanced_example():
    """Test enhanced prediction with example data (GET endpoint for browser)"""
    try:
        example_data = LoanApplication(
            Gender="Male",
            Married="Yes", 
            Dependents="1", 
            Education="Graduate",
            Self_Employed="No",
            ApplicantIncome=5849,
            CoapplicantIncome=0,
            LoanAmount=120,
            Loan_Amount_Term=360,
            Credit_History=1,
            Property_Area="Urban"
        )
        
        return await predict_loan_enhanced(example_data)
    except Exception as e:
        logger.error(f"Enhanced example error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/example/scenarios")
async def get_example_scenarios():
    """Show different loan scenarios for testing"""
    scenarios = [
        {
            "name": "High Income Graduate",
            "url": "/example/high-income",
            "description": "High income applicant with good credit"
        },
        {
            "name": "Low Income Risk",
            "url": "/example/low-income", 
            "description": "Low income applicant with risk factors"
        },
        {
            "name": "Medium Income",
            "url": "/example/medium-income",
            "description": "Average income applicant"
        }
    ]
    
    return {
        "available_scenarios": scenarios,
        "enhanced_versions": [scenario["url"] + "/enhanced" for scenario in scenarios]
    }

@app.get("/example/high-income")
async def high_income_example():
    """High income loan example"""
    example_data = LoanApplication(
        Gender="Male",
        Married="Yes",
        Dependents="2", 
        Education="Graduate",
        Self_Employed="No",
        ApplicantIncome=8000,
        CoapplicantIncome=3000,
        LoanAmount=200,
        Loan_Amount_Term=360,
        Credit_History=1,
        Property_Area="Urban"
    )
    return await predict_loan_basic(example_data)

@app.get("/example/high-income/enhanced")
async def high_income_enhanced_example():
    """High income loan example with AI explanation"""
    example_data = LoanApplication(
        Gender="Male",
        Married="Yes",
        Dependents="2",
        Education="Graduate", 
        Self_Employed="No",
        ApplicantIncome=8000,
        CoapplicantIncome=3000,
        LoanAmount=200,
        Loan_Amount_Term=360,
        Credit_History=1,
        Property_Area="Urban"
    )
    return await predict_loan_enhanced(example_data)

@app.get("/example/low-income")
async def low_income_example():
    """Low income loan example"""
    example_data = LoanApplication(
        Gender="Female",
        Married="No",
        Dependents="3+",
        Education="Not Graduate",
        Self_Employed="Yes", 
        ApplicantIncome=2000,
        CoapplicantIncome=0,
        LoanAmount=300,
        Loan_Amount_Term=180,
        Credit_History=0,
        Property_Area="Rural"
    )
    return await predict_loan_basic(example_data)

@app.get("/example/low-income/enhanced")
async def low_income_enhanced_example():
    """Low income loan example with AI explanation"""
    example_data = LoanApplication(
        Gender="Female",
        Married="No", 
        Dependents="3+",
        Education="Not Graduate",
        Self_Employed="Yes",
        ApplicantIncome=2000,
        CoapplicantIncome=0,
        LoanAmount=300,
        Loan_Amount_Term=180,
        Credit_History=0,
        Property_Area="Rural"
    )
    return await predict_loan_enhanced(example_data)

# POST endpoints (original functionality)
@app.post("/predict", response_model=PredictionResponse)
async def predict_loan_basic(loan_application: LoanApplication):
    """Basic loan prediction"""
    try:
        if predictor is None or predictor.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not available"
            )
        
        # Make prediction
        prediction_result = predictor.predict(loan_application)
        
        # Add metadata
        prediction_result["request_id"] = f"basic_req_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        prediction_result["timestamp"] = datetime.utcnow()
        
        return PredictionResponse(**prediction_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Basic prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction service error"
        )

@app.post("/predict/enhanced", response_model=EnhancedPredictionResponse)
async def predict_loan_enhanced(loan_application: LoanApplication):
    """Enhanced loan prediction with AI explanations"""
    start_time = datetime.utcnow()
    
    try:
        if predictor is None or predictor.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model not available"
            )
        
        logger.info(f"Enhanced prediction request for applicant income: {loan_application.ApplicantIncome}")
        
        # Make ML prediction
        prediction_result = predictor.predict(loan_application)
        
        # Generate AI explanation
        if LLM_AVAILABLE:
            ai_explanation = generate_enhanced_explanation(
                prediction_result, 
                loan_application.dict()
            )
        else:
            # Fallback explanation
            ai_explanation = {
                "explanation": f"This loan was {prediction_result['prediction_label'].lower()} based on the applicant's financial profile and risk assessment.",
                "advice": ["Maintain good financial habits", "Monitor your credit score regularly"],
                "llm_available": False,
                "generated_at": datetime.utcnow().isoformat(),
                "model_type": "basic"
            }
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Combine results
        enhanced_result = {
            **prediction_result,
            "ai_explanation": ai_explanation,
            "request_id": f"enhanced_req_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.utcnow(),
            "processing_time_ms": processing_time
        }
        
        logger.info(f"Enhanced prediction completed: {prediction_result['prediction_label']} "
                   f"(Processing: {processing_time:.1f}ms)")
        
        return EnhancedPredictionResponse(**enhanced_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Enhanced prediction service error"
        )

@app.get("/model/info")
async def get_model_info():
    """Get comprehensive model information"""
    try:
        ml_info = predictor.get_model_info() if predictor else {"status": "not_available"}
        
        info = {
            "ml_model": ml_info,
            "explanation_system": {
                "available": LLM_AVAILABLE,
                "type": "rule-based" if LLM_AVAILABLE else "basic",
                "features": [
                    "Intelligent decision explanations",
                    "Risk factor analysis", 
                    "Personalized advice"
                ] if LLM_AVAILABLE else ["Basic explanations"]
            },
            "api_version": "2.0.0",
            "endpoints": {
                "GET (browser-friendly)": [
                    "/ - API information",
                    "/example - Basic prediction test",
                    "/example/enhanced - Enhanced prediction test", 
                    "/example/scenarios - Different test scenarios"
                ],
                "POST (programmatic)": [
                    "/predict - Basic prediction",
                    "/predict/enhanced - Enhanced prediction"
                ]
            },
            "timestamp": datetime.utcnow()
        }
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve model information"
        )

@app.get("/llm/status")
async def llm_status():
    """Get LLM status"""
    return {
        "llm_available": LLM_AVAILABLE,
        "model_type": "rule-based" if LLM_AVAILABLE else "none",
        "features": [
            "Intelligent explanations",
            "Risk analysis",
            "Personalized advice", 
            "Fast CPU inference"
        ] if LLM_AVAILABLE else [],
        "timestamp": datetime.utcnow()
    }

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )