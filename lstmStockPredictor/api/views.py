from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from .LSTMStockPredictor import LSTMStockPredictor  # Import your model class

class PredictStock(APIView):
    def post(self, request):
        company = request.data.get('company')
        
        # Initialize and run the model
        predictor = LSTMStockPredictor(company=company)
        predictor.fetch_data()
        predictor.scale_data()
        predictor.train_model()
        results = predictor.get_results()
        
        # Return results as a JSON response
        if results is not None:
            results_dict = results.to_dict(orient='records')
            return Response({"predictions": results_dict}, status=200)
        else:
            return Response({"error": "Failed to generate predictions"}, status=500)
