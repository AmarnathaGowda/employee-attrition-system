# core/serializers.py
from rest_framework import serializers
from .models import Employee, Prediction, RetentionAction

class EmployeeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Employee
        fields = '__all__'

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'

class RetentionActionSerializer(serializers.ModelSerializer):
    class Meta:
        model = RetentionAction
        fields = '__all__'
