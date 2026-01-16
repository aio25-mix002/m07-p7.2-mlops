#!/usr/bin/env bash
set -e

echo "🌐 Port-forwarding MLflow UI..."
echo "   Namespace: mlops"
echo "   Service:   mlflow"
echo "   Local URL: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop."

kubectl port-forward svc/mlflow -n mlops 5000:5000