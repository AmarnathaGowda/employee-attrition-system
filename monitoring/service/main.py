# monitoring/service/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
from datetime import datetime
from monitoring.middleware import MonitoringService

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="API for predicting employee attrition risk using ML models",
    version="1.0.0",
    contact={
        "name": "Your Engineering Team",
        "email": "ai-team@company.com"
    },
    openapi_tags=[
        {
            "name": "Predictions",
            "description": "Endpoints for making attrition risk predictions"
        },
        {
            "name": "Monitoring",
            "description": "Model performance monitoring and data drift detection"
        }
    ]
)

monitor = MonitoringService()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class PredictionInput(BaseModel):
    employee_id: str = "550e8400-e29b-41d4-a716-446655440000"
    age: int = 35
    monthly_income: float = 6500.0
    total_working_years: int = 7
    department: str = "Research & Development"
    education_field: str = "Life Sciences"
    job_role: str = "Research Scientist"

class Config:
        schema_extra = {
            "example": {
                "employee_id": "550e8400-e29b-41d4-a716-446655440000",
                "age": 35,
                "monthly_income": 6500,
                "total_working_years": 7,
                "department": "Research & Development",
                "education_field": "Life Sciences",
                "job_role": "Research Scientist"
            }
        }

@app.post("/predict", 
         tags=["Predictions"],
         summary="Predict attrition risk",
         response_description="Predicted attrition risk score")
async def predict_attrition(input: PredictionInput):
    try:
        """
    Predicts the attrition risk for an employee based on their profile
    
    - **employee_id**: Unique employee identifier (UUID format)
    - **age**: Employee's age in years
    - **monthly_income**: Gross monthly salary
    - **total_working_years**: Total years of professional experience
    - **department**: Current department
    - **education_field**: Field of education
    - **job_role**: Current job role
    """
        # Load model pipeline
        model = joblib.load("ml_engine/models/xgboost_pipeline.joblib")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input.dict()])
        
        # Get prediction
        risk_score = model.predict_proba(input_df)[0][1]
        
        # Log prediction
        logger.info(f"Prediction for {input.employee_id}: {risk_score:.4f}")

        # Log to monitoring
        monitor.log_prediction({
            'employee_id': input.employee_id,
            'age': input.age,
            'monthly_income': input.monthly_income,
            'department': input.department,
            'job_role': input.j_role
        })
        
        return {
            "employee_id": input.employee_id,
            "risk_score": round(risk_score, 4),
            "timestamp": datetime.utcnow().isoformat()
        }
    
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/monitor",
        tags=["Monitoring"],
        summary="Model monitoring report",
        description="Returns data drift and model performance metrics")
async def get_monitoring_report():
    report = monitor.generate_monitoring_report()
    if report:
        return report.json()
    raise HTTPException(status_code=404, detail="Report unavailable")
