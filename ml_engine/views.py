# ml_engine/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .model_registry import ModelRegistry
from .serializers import PredictionInputSerializer


registry = ModelRegistry()

class PredictionView(APIView):
    def post(self, request):
        serializer = PredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        try:
            risk_score = registry.predict_proba(
                data['model_version'],
                [data['input_data']]
            )[0]
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response({
            'employee_id': str(data['employee_id']),
            'model_version': data['model_version'],
            'attrition_risk': round(risk_score, 4)
        })
