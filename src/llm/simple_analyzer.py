# src/llm/simple_analyzer.py - Simplified LLM implementation for CPU
import logging
from typing import Dict, Any, List
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class SimpleLoanExplainer:
    """Simplified loan explainer using rule-based generation"""
    
    def __init__(self):
        self.is_ready = True
        self.model_name = "rule-based-explainer"
    
    def is_available(self) -> bool:
        return self.is_ready
    
    def generate_loan_explanation(self, prediction_result: Dict[str, Any], loan_application: Dict[str, Any]) -> str:
        """Generate intelligent explanation using business rules"""
        
        decision = prediction_result['prediction_label']
        confidence = prediction_result['confidence']
        risk_factors = prediction_result['risk_factors']
        
        # Extract key details
        total_income = loan_application['ApplicantIncome'] + loan_application['CoapplicantIncome']
        loan_amount = loan_application['LoanAmount']
        debt_ratio = risk_factors['debt_to_income_ratio']
        credit_history = risk_factors['credit_history']
        education = loan_application['Education']
        
        if decision == "Approved":
            return self._generate_approval_explanation(
                confidence, total_income, loan_amount, debt_ratio, credit_history, education
            )
        else:
            return self._generate_rejection_explanation(
                confidence, total_income, loan_amount, debt_ratio, credit_history, risk_factors['high_risk_factors']
            )
    
    def _generate_approval_explanation(self, confidence, income, loan_amount, debt_ratio, credit_history, education):
        """Generate approval explanation"""
        
        reasons = []
        
        if credit_history == "Good":
            reasons.append("excellent credit history demonstrating reliable payment behavior")
        
        if debt_ratio < 4:
            reasons.append("conservative debt-to-income ratio indicating strong repayment capacity")
        elif debt_ratio < 6:
            reasons.append("manageable debt-to-income ratio within acceptable lending standards")
        
        if income > 8000:
            reasons.append("high income level providing strong financial stability")
        elif income > 5000:
            reasons.append("solid income foundation supporting loan repayment")
        
        if education == "Graduate":
            reasons.append("graduate education often correlating with stable career prospects")
        
        primary_reason = reasons[0] if reasons else "favorable overall financial profile"
        
        explanation = f"This loan application received approval with {confidence:.1%} confidence primarily due to {primary_reason}"
        
        if len(reasons) > 1:
            additional = ", ".join(reasons[1:2])  # Add one more reason
            explanation += f", complemented by {additional}"
        
        explanation += f". With a total household income of ${income:,} and a debt-to-income ratio of {debt_ratio:.1f}, the applicant demonstrates low financial risk for a ${loan_amount}k loan."
        
        return explanation
    
    def _generate_rejection_explanation(self, confidence, income, loan_amount, debt_ratio, credit_history, high_risk_factors):
        """Generate rejection explanation"""
        
        primary_concerns = []
        
        if credit_history == "Poor":
            primary_concerns.append("poor credit history indicating past payment difficulties")
        
        if debt_ratio > 8:
            primary_concerns.append("extremely high debt-to-income ratio of {:.1f} creating significant repayment risk".format(debt_ratio))
        elif debt_ratio > 6:
            primary_concerns.append("elevated debt-to-income ratio of {:.1f} exceeding prudent lending thresholds".format(debt_ratio))
        
        if income < 3000:
            primary_concerns.append("insufficient income level relative to the requested loan amount")
        
        main_concern = primary_concerns[0] if primary_concerns else "multiple risk factors identified during assessment"
        
        explanation = f"This loan application was declined with {confidence:.1%} confidence due to {main_concern}"
        
        if len(primary_concerns) > 1:
            explanation += f", combined with {primary_concerns[1]}"
        
        explanation += f". The analysis indicates potential challenges in servicing a ${loan_amount}k loan with the current financial profile."
        
        return explanation
    
    def generate_risk_advice(self, risk_factors: Dict[str, Any]) -> List[str]:
        """Generate personalized advice"""
        advice = []
        
        if risk_factors['credit_history'] == "Poor":
            advice.extend([
                "Focus on improving your credit score by paying all bills on time",
                "Consider paying down existing debts to improve your credit utilization ratio"
            ])
        
        if risk_factors['debt_to_income_ratio'] > 6:
            advice.extend([
                "Reduce your debt-to-income ratio by increasing income or paying down debts", 
                "Consider applying for a smaller loan amount that better fits your income level"
            ])
        elif risk_factors['debt_to_income_ratio'] > 4:
            advice.append("Maintain or improve your current debt-to-income ratio before applying for larger loans")
        
        if risk_factors['risk_level'] == "High":
            advice.append("Build a stronger financial foundation with steady employment and emergency savings")
        
        if not advice:
            advice.extend([
                "Your financial profile is strong - continue maintaining good financial habits",
                "Consider building additional savings for future financial goals"
            ])
        
        return advice[:3]  # Limit to top 3 pieces of advice
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_type": "Rule-based Loan Analysis Engine",
            "available": True,
            "device": "cpu",
            "features": [
                "Intelligent decision explanations",
                "Risk factor analysis", 
                "Personalized financial advice",
                "Fast CPU inference"
            ]
        }

# Singleton instance
_simple_explainer = None

def get_simple_loan_explainer() -> SimpleLoanExplainer:
    """Get singleton explainer instance"""
    global _simple_explainer
    if _simple_explainer is None:
        _simple_explainer = SimpleLoanExplainer()
    return _simple_explainer

def generate_enhanced_explanation(prediction_result: Dict[str, Any], loan_application: Dict[str, Any]) -> Dict[str, Any]:
    """Generate enhanced explanation using simple explainer"""
    explainer = get_simple_loan_explainer()
    
    explanation = explainer.generate_loan_explanation(prediction_result, loan_application)
    advice = explainer.generate_risk_advice(prediction_result['risk_factors'])
    
    return {
        "explanation": explanation,
        "advice": advice,
        "llm_available": True,  # Our rule-based system is always available
        "generated_at": datetime.utcnow().isoformat(),
        "model_type": "rule-based"
    }
