# monitoring/middleware.py
from evidently.report import Report
from evidently.metrics import *
import pandas as pd
import logging

class MonitoringService:
    def __init__(self):
        self.reference_data = pd.read_csv("ml_engine/data/emp_attrition.csv")
        self.current_data = []
        
    def log_prediction(self, data):
        self.current_data.append(data)
        
    def generate_monitoring_report(self):
        try:
            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset(),
                ClassificationPreset()
            ])
            
            current_df = pd.DataFrame(self.current_data[-1000:])  # Last 1000 predictions
            report.run(
                reference_data=self.reference_data, 
                current_data=current_df
            )
            
            return report
            
        except Exception as e:
            logging.error(f"Monitoring failed: {str(e)}")
            return None
