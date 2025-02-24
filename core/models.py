# core/models.py
from django.db import models
import uuid

class Employee(models.Model):
    EMPLOYMENT_CHOICES = [
        ('FT', 'Full-time'),
        ('PT', 'Part-time'),
        ('CN', 'Contract')
    ]
    
    # id = models.UUIDField(primary_key=True, editable=False)
    id =  models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,  # Add automatic UUID generation
        editable=False,
        unique=True
    )
    name = models.CharField(max_length=255)
    age = models.IntegerField()
    department = models.CharField(max_length=100)
    job_role = models.CharField(max_length=100)
    employment_type = models.CharField(max_length=2, choices=EMPLOYMENT_CHOICES)
    satisfaction_score = models.FloatField()
    last_evaluation = models.DateTimeField(auto_now_add=True)
    attrition_risk = models.FloatField(null=True, blank=True)

class Prediction(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    model_version = models.CharField(max_length=20)
    prediction_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

class RetentionAction(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    action_type = models.CharField(max_length=100)
    implementation_date = models.DateField()
    effectiveness = models.FloatField(null=True, blank=True)
