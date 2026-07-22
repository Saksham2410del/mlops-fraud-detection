"""
MLOps Fraud Detection Pipeline
================================

A production-grade machine learning pipeline for detecting fraudulent
transactions using the PaySim synthetic dataset.

Modules:
    preprocess: Data loading, cleaning, feature engineering, and train/test splitting.
    train:      XGBoost model training with MLflow experiment tracking.
    evaluate:   Comprehensive model evaluation with business metrics.
    drift:      Data drift detection using Evidently for monitoring.
"""
