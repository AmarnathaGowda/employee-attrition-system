# monitoring/service/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
from datetime import datetime
from monitoring.middleware import MonitoringService

app = FastAPI()

monitor = MonitoringService()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class PredictionInput(BaseModel):
    employee_id: str
    age: int
    monthly_income: float
    total_working_years: int
    department: str
    education_field: str
    job_role: str

@app.post("/predict")
async def predict_attrition(input: PredictionInput):
    try:
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
    
@app.get("/monitor")
async def get_monitoring_report():
    report = monitor.generate_monitoring_report()
    if report:
        return report.json()
    raise HTTPException(status_code=404, detail="Report unavailable")
