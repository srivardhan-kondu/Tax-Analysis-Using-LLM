"""
AdaBoost Classifier for Tax Evasion Detection
Implements scikit-learn AdaBoostClassifier with probability calibration
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import joblib
import logging

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaxEvasionAdaBoost:
    """
    AdaBoost classifier for tax evasion risk prediction
    
    Features:
    - Uses DecisionTree as base estimator
    - Probability calibration for accurate confidence scores
    - Implements feature engineering for financial ratios
    - Provides probability estimates for each risk level
    - Supports model persistence (save/load)
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.8):
        """
        Initialize the AdaBoost classifier
        
        Args:
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate shrinks contribution of each classifier
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
        # Base estimator: Decision Tree with controlled depth
        base_estimator = DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        # Base LLM 
        self.base_model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        
        # Calibrated model for better probability estimates
        self.model = None  # Will be set after calibration
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'sales', 'profit_margin', 'tax_to_revenue_ratio',
            'revenue_growth', 'employee_growth', 'debt_ratio', 'operating_expenses'
        ]
        self.class_labels = ['No Risk', 'Medium Risk', 'High Risk']
        
        # Training history
        self.training_accuracy = None
        self.test_accuracy = None
        self.cv_scores = None
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features for better prediction
        
        Args:
            df: DataFrame with raw financial features
            
        Returns:
            DataFrame with additional engineered features
        """
        df = df.copy()
        
        # Feature 1: Profit-to-Tax Ratio (higher = more suspicious)
        df['profit_tax_ratio'] = df['profit_margin'] / (df['tax_to_revenue_ratio'] + 0.001)
        
        # Feature 2: Growth Discrepancy (revenue vs employee growth)
        df['growth_discrepancy'] = df['revenue_growth'] - df['employee_growth']
        
        # Feature 3: Operating Efficiency
        df['operating_efficiency'] = df['operating_expenses'] / (df['sales'] + 1)
        
        # Feature 4: Tax Efficiency Score (lower = more suspicious)
        df['tax_efficiency'] = df['tax_to_revenue_ratio'] / (df['profit_margin'] + 0.001)
        
        return df
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare and scale features for the model
        
        Args:
            data: DataFrame with company financial data
            
        Returns:
            Scaled feature array
        """
        # Ensure all required columns exist
        for col in self.feature_names:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Engineer additional features
        data_engineered = self._engineer_features(data[self.feature_names])
        
        # All features (original + engineered)
        all_features = self.feature_names + [
            'profit_tax_ratio', 'growth_discrepancy', 
            'operating_efficiency', 'tax_efficiency'
        ]
        
        return data_engineered[all_features].values
    
    def train(self, X: pd.DataFrame, y: np.ndarray, 
              test_size: float = 0.2) -> Dict:
        """
        Train the LLM  with probability calibration
        
        Args:
            X: DataFrame with financial features
            y: Array of risk labels (0, 1, 2)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training...")
        
        # Prepare features
        X_features = self.prepare_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train calibrated model with cross-validation
        # This produces better probability estimates
        self.model = CalibratedClassifierCV(
            self.base_model, 
            method='sigmoid',  # Works well with boosting methods
            cv=3  # 3-fold cross-validation for calibration
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Store base model for feature importance
        self.base_model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Calculate accuracies
        self.training_accuracy = self.base_model.score(X_train_scaled, y_train)
        self.test_accuracy = self.model.score(X_test_scaled, y_test)
        
        # Cross-validation scores on training data
        self.cv_scores = cross_val_score(
            self.base_model, X_train_scaled, y_train, cv=5, scoring='accuracy'
        )
        
        logger.info(f"Training Accuracy: {self.training_accuracy:.4f}")
        logger.info(f"Test Accuracy: {self.test_accuracy:.4f}")
        logger.info(f"CV Score (mean): {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std() * 2:.4f})")
        
        return {
            'training_accuracy': self.training_accuracy,
            'test_accuracy': self.test_accuracy,
            'cv_scores': self.cv_scores,
            'cv_mean': self.cv_scores.mean(),
            'cv_std': self.cv_scores.std(),
            'X_test': X_test_scaled,
            'y_test': y_test,
            'X_train': X_train_scaled,
            'y_train': y_train
        }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict risk levels for companies
        
        Args:
            data: DataFrame with company financial data
            
        Returns:
            Array of predicted risk levels (0, 1, 2)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_features = self.prepare_features(data)
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get calibrated probability estimates for each risk level
        
        Args:
            data: DataFrame with company financial data
            
        Returns:
            Array of shape (n_samples, 3) with calibrated probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_features = self.prepare_features(data)
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict_proba(X_scaled)
    
    def predict_single(self, company_data: Dict) -> Dict:
        """
        Predict risk for a single company with detailed output
        
        Args:
            company_data: Dictionary with company financial data
            
        Returns:
            Dictionary with prediction details
        """
        df = pd.DataFrame([company_data])
        prediction = self.predict(df)[0]
        probabilities = self.predict_proba(df)[0]
        
        return {
            'risk_level': int(prediction),
            'risk_category': self.class_labels[prediction],
            'confidence': float(max(probabilities)),
            'probabilities': {
                'No Risk': float(probabilities[0]),
                'Medium Risk': float(probabilities[1]),
                'High Risk': float(probabilities[2])
            }
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from base model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # All feature names (original + engineered)
        all_feature_names = self.feature_names + [
            'profit_tax_ratio', 'growth_discrepancy',
            'operating_efficiency', 'tax_efficiency'
        ]
        
        importances = self.base_model.feature_importances_
        
        return dict(sorted(
            zip(all_feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        ))
    
    def save_model(self, filepath: str = None) -> str:
        """Save trained model to file"""
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'adaboost_model.joblib'
            )
        
        model_data = {
            'model': self.model,
            'base_model': self.base_model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy,
            'test_accuracy': self.test_accuracy,
            'cv_scores': self.cv_scores,
            'feature_names': self.feature_names,
            'class_labels': self.class_labels
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to: {filepath}")
        return filepath
    
    def load_model(self, filepath: str = None) -> bool:
        """Load trained model from file"""
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'adaboost_model.joblib'
            )
        
        if not os.path.exists(filepath):
            logger.warning(f"Model file not found: {filepath}")
            return False
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.base_model = model_data.get('base_model', model_data['model'])
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.training_accuracy = model_data['training_accuracy']
        self.test_accuracy = model_data['test_accuracy']
        self.cv_scores = model_data['cv_scores']
        self.feature_names = model_data['feature_names']
        self.class_labels = model_data['class_labels']
        
        logger.info(f"Model loaded from: {filepath}")
        return True


if __name__ == "__main__":
    # Quick test
    from training_data import generate_training_data, get_feature_columns
    
    print("Generating training data...")
    df = generate_training_data(1000)
    
    print("Training LLM  with probability calibration...")
    model = TaxEvasionAdaBoost(n_estimators=100, learning_rate=0.8)
    
    X = df[get_feature_columns()]
    y = df['risk_label'].values
    
    results = model.train(X, y)
    
    print(f"\n✅ Training Complete!")
    print(f"Training Accuracy: {results['training_accuracy']:.2%}")
    print(f"Test Accuracy: {results['test_accuracy']:.2%}")
    print(f"CV Score: {results['cv_mean']:.2%} (+/- {results['cv_std']*2:.2%})")
    
    print("\nFeature Importance:")
    for feature, importance in model.get_feature_importance().items():
        print(f"  {feature}: {importance:.4f}")
    
    # Save model
    model.save_model()
    print("\n✅ Model saved!")
