# ml_engine/serializers.py
from rest_framework import serializers

class PredictionInputSerializer(serializers.Serializer):
    employee_id = serializers.UUIDField()
    model_version = serializers.ChoiceField(choices=['logreg', 'randomforest', 'xgboost'])
    input_data = serializers.JSONField()
