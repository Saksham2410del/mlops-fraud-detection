"""
MLOps Fraud Detection Pipeline

Modules:
    preprocess - Data loading, cleaning, feature engineering, SMOTE
    train      - XGBoost training with MLflow tracking
    evaluate   - Model evaluation and threshold tuning
    predict    - FastAPI inference server
    explain    - SHAP + LLM explainability
    drift      - Data drift detection with Evidently
"""
