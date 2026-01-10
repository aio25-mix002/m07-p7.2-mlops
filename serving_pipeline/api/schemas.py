from pydantic import BaseModel, Field
from typing import Literal


class ChurnInput(BaseModel):
    """Input schema for churn prediction"""
    
    Age: int = Field(..., ge=18, le=65, description="Customer age")
    Gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    Tenure: int = Field(..., ge=1, le=60, description="Months with company")
    Usage_Frequency: int = Field(..., ge=1, le=30, description="Monthly usage frequency")
    Support_Calls: int = Field(..., ge=0, le=10, description="Number of support calls")
    Payment_Delay: int = Field(..., ge=0, le=30, description="Payment delay in days")
    Subscription_Type: Literal["Basic", "Standard", "Premium"] = Field(..., description="Subscription tier")
    Contract_Length: Literal["Monthly", "Quarterly", "Annual"] = Field(..., description="Contract duration")
    Total_Spend: float = Field(..., ge=100, le=1000, description="Total amount spent")
    Last_Interaction: int = Field(..., ge=1, le=30, description="Days since last interaction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Age": 30,
                "Gender": "Female",
                "Tenure": 39,
                "Usage_Frequency": 14,
                "Support_Calls": 5,
                "Payment_Delay": 18,
                "Subscription_Type": "Standard",
                "Contract_Length": "Annual",
                "Total_Spend": 932.0,
                "Last_Interaction": 17
            }
        }


class ChurnPrediction(BaseModel):
    """Prediction response"""
    churn: int = Field(..., description="Predicted churn (0=Active, 1=Churn)")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str

