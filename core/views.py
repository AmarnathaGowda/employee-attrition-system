# core/views.py
from rest_framework import viewsets
from .models import Employee, Prediction, RetentionAction
from .serializers import EmployeeSerializer, PredictionSerializer, RetentionActionSerializer

class EmployeeViewSet(viewsets.ModelViewSet):
    queryset = Employee.objects.all()
    serializer_class = EmployeeSerializer

class PredictionViewSet(viewsets.ModelViewSet):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer

class RetentionActionViewSet(viewsets.ModelViewSet):
    queryset = RetentionAction.objects.all()
    serializer_class = RetentionActionSerializer
