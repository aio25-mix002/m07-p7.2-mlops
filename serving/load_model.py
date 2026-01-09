import mlflow
import os
import pandas as pd


def load_model(model_uri: str = "runs:/c4b92406479d490993622563a35a47f7/xgboost_churn"):
    """
    Load model from MLflow
    
    Args:
        model_uri: MLflow model URI (default: latest model)
    
    Returns:
        Loaded MLflow model
    """
    # 1. Configure MLflow Tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # 2. Configure S3/MinIO Credentials
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    
    # 3. Load model
    model = mlflow.pyfunc.load_model(model_uri)
    
    print("Model loaded successfully!")
    
    return model


if __name__ == "__main__":
    # Test loading model
    model = load_model()
    
    # Get feature names
    custom_model_instance = model.unwrap_python_model()
    print("Feature names:", custom_model_instance.feature_names)
