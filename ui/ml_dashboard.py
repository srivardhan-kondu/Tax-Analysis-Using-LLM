"""
ML Dashboard UI for Tax Evasion Detection System
Auto-trains on 1000 companies and displays evaluation metrics
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.adaboost_model import TaxEvasionAdaBoost
from ml.ml_evaluator import ModelEvaluator
from ml.training_data import generate_training_data, get_feature_columns

# Fixed optimal hyperparameters (no user adjustment needed)
N_ESTIMATORS = 100
LEARNING_RATE = 0.8
N_TRAINING_SAMPLES = 10000  # 10K samples for robust training


def get_or_train_model():
    """
    Get existing model or auto-train if not available.
    Model is trained on 1000 companies with optimal hyperparameters.
    """
    # Return cached if BOTH model and metrics exist
    if (st.session_state.get('ml_model') is not None and 
        st.session_state.get('ml_trained') and 
        st.session_state.get('ml_metrics') is not None):
        return st.session_state.ml_model, st.session_state.ml_metrics
    
    model_path = os.path.join(project_root, 'data', 'adaboost_model.joblib')
    model = TaxEvasionAdaBoost(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
    
    # Try to load existing model
    if os.path.exists(model_path):
        if model.load_model():
            st.session_state.ml_model = model
            st.session_state.ml_trained = True
            
            # Generate metrics for display
            df = generate_training_data(N_TRAINING_SAMPLES)
            X = df[get_feature_columns()]
            y = df['risk_label'].values
            results = model.train(X, y)
            
            y_pred = model.model.predict(results['X_test'])
            y_proba = model.model.predict_proba(results['X_test'])
            
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(results['y_test'], y_pred, y_proba)
            metrics['feature_importance'] = model.get_feature_importance()
            metrics['cv_scores'] = results['cv_scores']
            st.session_state.ml_metrics = metrics
            
            return model, metrics
    
    # Auto-train new model
    return auto_train_model()


def auto_train_model():
    """Auto-train model on 1000 companies with fixed optimal parameters"""
    
    with st.spinner("🤖 Auto-training LLM  on 1000 companies..."):
        # Generate training data
        df = generate_training_data(N_TRAINING_SAMPLES)
        X = df[get_feature_columns()]
        y = df['risk_label'].values
        
        # Train model
        model = TaxEvasionAdaBoost(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
        results = model.train(X, y)
        
        # Calculate metrics
        y_pred = model.model.predict(results['X_test'])
        y_proba = model.model.predict_proba(results['X_test'])
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(results['y_test'], y_pred, y_proba)
        metrics['feature_importance'] = model.get_feature_importance()
        metrics['cv_scores'] = results['cv_scores']
        
        # Save model
        model.save_model()
        
        # Store in session state
        st.session_state.ml_model = model
        st.session_state.ml_metrics = metrics
        st.session_state.ml_trained = True
        
        return model, metrics


def show_ml_dashboard():
    """Display the ML Model dashboard with auto-trained model and evaluation metrics"""
    
    st.title("🤖 AdaBoost ML Model")
    st.markdown("---")
    
    # Initialize session state
    if 'ml_model' not in st.session_state:
        st.session_state.ml_model = None
    if 'ml_metrics' not in st.session_state:
        st.session_state.ml_metrics = None
    if 'ml_trained' not in st.session_state:
        st.session_state.ml_trained = False
    
    # Auto-load or train model
    model, metrics = get_or_train_model()
    
    # Model info
    st.success(f"✅ Model Ready! Trained on **{N_TRAINING_SAMPLES} companies** with optimal settings.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", f"{N_TRAINING_SAMPLES:,}")
    with col2:
        st.metric("Test Accuracy", f"{metrics['accuracy']:.2%}")
    with col3:
        st.metric("F1-Score", f"{metrics['f1_macro']:.2%}")
    with col4:
        auc_val = metrics.get('auc_macro', 0)
        st.metric("AUC", f"{auc_val:.2%}")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "📈 Visualizations", "ℹ️ About AdaBoost"])
    
    with tab1:
        show_performance_metrics(metrics)
    
    with tab2:
        show_visualizations(metrics)
    
    with tab3:
        show_about_adaboost()


def show_performance_metrics(metrics):
    """Display performance metrics table"""
    
    st.subheader("📊 Performance Metrics")
    
    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{metrics['precision_macro']:.2%}")
    with col3:
        st.metric("Recall", f"{metrics['recall_macro']:.2%}")
    with col4:
        st.metric("F1-Score", f"{metrics['f1_macro']:.2%}")
    with col5:
        auc_val = metrics.get('auc_macro', 0)
        st.metric("AUC", f"{auc_val:.2%}")
    
    st.markdown("---")
    
    # Classification Report
    st.subheader("📋 Classification Report")
    report = metrics['classification_report']
    
    report_df = pd.DataFrame({
        'Class': ['No Risk', 'Medium Risk', 'High Risk', 'Macro Avg', 'Weighted Avg'],
        'Precision': [
            report['No Risk']['precision'],
            report['Medium Risk']['precision'],
            report['High Risk']['precision'],
            report['macro avg']['precision'],
            report['weighted avg']['precision']
        ],
        'Recall': [
            report['No Risk']['recall'],
            report['Medium Risk']['recall'],
            report['High Risk']['recall'],
            report['macro avg']['recall'],
            report['weighted avg']['recall']
        ],
        'F1-Score': [
            report['No Risk']['f1-score'],
            report['Medium Risk']['f1-score'],
            report['High Risk']['f1-score'],
            report['macro avg']['f1-score'],
            report['weighted avg']['f1-score']
        ],
        'Support': [
            int(report['No Risk']['support']),
            int(report['Medium Risk']['support']),
            int(report['High Risk']['support']),
            int(report['macro avg']['support']),
            int(report['weighted avg']['support'])
        ]
    })
    
    st.dataframe(report_df, use_container_width=True, hide_index=True)
    
    # Cross-validation
    if 'cv_scores' in metrics:
        st.subheader("🔄 Cross-Validation (5-Fold)")
        cv_scores = metrics['cv_scores']
        
        col1, col2 = st.columns([3, 1])
        with col1:
            cv_df = pd.DataFrame({
                'Fold': [f'Fold {i+1}' for i in range(len(cv_scores))],
                'Accuracy': cv_scores
            })
            st.bar_chart(cv_df.set_index('Fold'))
        with col2:
            st.metric("Mean", f"{cv_scores.mean():.2%}")
            st.metric("Std Dev", f"±{cv_scores.std():.2%}")


def show_visualizations(metrics):
    """Display visualizations"""
    
    evaluator = ModelEvaluator()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy Gauge
        st.subheader("🎯 Model Accuracy")
        fig_gauge = evaluator.plot_accuracy_gauge(metrics['accuracy'])
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Confusion Matrix
        st.subheader("📊 Confusion Matrix")
        fig_cm = evaluator.plot_confusion_matrix(metrics['confusion_matrix'])
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # Per-class metrics
        st.subheader("📈 Per-Class Performance")
        fig_metrics = evaluator.plot_metrics_comparison(metrics)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Feature Importance
        if 'feature_importance' in metrics:
            st.subheader("🎯 Feature Importance")
            fig_fi = evaluator.plot_feature_importance(metrics['feature_importance'])
            st.plotly_chart(fig_fi, use_container_width=True)
    
    # ROC Curves (full width)
    if 'fpr' in metrics and 'tpr' in metrics:
        st.subheader("📈 ROC Curves")
        fig_roc = evaluator.plot_roc_curves(metrics['fpr'], metrics['tpr'], metrics['roc_auc'])
        st.plotly_chart(fig_roc, use_container_width=True)


def show_about_adaboost():
    """Show information about AdaBoost algorithm"""
    
    st.subheader("ℹ️ About AdaBoost Algorithm")
    
    st.markdown("""
    ### What is AdaBoost?
    
    **AdaBoost** (Adaptive Boosting) is a powerful ensemble learning algorithm that combines 
    multiple weak classifiers to create a strong classifier.
    
    ### How It Works
    
    1. **Initialize**: Assign equal weights to all training samples
    2. **Train**: Build a weak classifier (Decision Tree) on weighted data
    3. **Evaluate**: Calculate classifier error rate
    4. **Update Weights**: Increase weights for misclassified samples
    5. **Repeat**: Train next classifier focusing on hard examples
    6. **Combine**: Final prediction is weighted vote of all classifiers
    
    ### Why AdaBoost for Tax Evasion Detection?
    
    | Advantage | Benefit |
    |-----------|---------|
    | **Handles Imbalanced Data** | Tax evasion cases are rare, AdaBoost focuses on them |
    | **Feature Importance** | Shows which financial factors matter most |
    | **Less Overfitting** | Ensemble approach is more robust |
    | **Interpretable** | Decision trees are easy to understand |
    
    ### Model Configuration
    
    Our model uses:
    - **100 Decision Tree estimators**
    - **Learning rate: 0.8**
    - **Max tree depth: 3** (prevents overfitting)
    - **Training data: 1000 synthetic companies**
    
    ### Training Data Distribution
    
    - **50% No Risk** (500 companies)
    - **30% Medium Risk** (300 companies)  
    - **20% High Risk** (200 companies)
    """)


if __name__ == "__main__":
    show_ml_dashboard()
