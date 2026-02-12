"""
ML Model Evaluator for Tax Evasion Detection
Provides comprehensive evaluation metrics and visualizations
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations
    
    Provides:
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix
    - ROC Curves and AUC
    - Classification Report
    - Feature Importance Visualization
    """
    
    def __init__(self, class_labels: list = None):
        """
        Initialize evaluator
        
        Args:
            class_labels: List of class names ['No Risk', 'Medium Risk', 'High Risk']
        """
        self.class_labels = class_labels or ['No Risk', 'Medium Risk', 'High Risk']
        self.n_classes = len(self.class_labels)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_proba: np.ndarray = None) -> Dict:
        """
        Compute all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional, for ROC)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_per_class': precision_score(y_true, y_pred, average=None, zero_division=0),
            'recall_per_class': recall_score(y_true, y_pred, average=None, zero_division=0),
            'f1_per_class': f1_score(y_true, y_pred, average=None, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(
                y_true, y_pred, 
                target_names=self.class_labels,
                output_dict=True,
                zero_division=0
            )
        }
        
        # ROC and AUC (if probabilities provided)
        if y_proba is not None:
            roc_data = self._compute_roc_auc(y_true, y_proba)
            metrics.update(roc_data)
        
        return metrics
    
    def _compute_roc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Compute ROC curves and AUC for multi-class"""
        # Binarize labels for ROC
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Micro-average
        fpr['micro'], tpr['micro'], _ = roc_curve(
            y_true_bin.ravel(), y_proba.ravel()
        )
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        
        # Macro-average
        roc_auc['macro'] = np.mean([roc_auc[i] for i in range(self.n_classes)])
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'auc_macro': roc_auc['macro']
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray) -> go.Figure:
        """
        Create confusion matrix heatmap
        
        Args:
            cm: Confusion matrix array
            
        Returns:
            Plotly figure
        """
        # Calculate percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create text annotations
        annotations = []
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                annotations.append(f"{cm[i][j]}<br>({cm_normalized[i][j]:.1f}%)")
        
        annotations = np.array(annotations).reshape(cm.shape)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=self.class_labels,
            y=self.class_labels,
            colorscale='Blues',
            text=annotations,
            texttemplate="%{text}",
            textfont={"size": 14},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        ))
        
        fig.update_layout(
            title={
                'text': '📊 Confusion Matrix',
                'font': {'size': 20}
            },
            xaxis_title='Predicted Label',
            yaxis_title='Actual Label',
            height=450,
            width=500,
            yaxis={'autorange': 'reversed'}
        )
        
        return fig
    
    def plot_roc_curves(self, fpr: Dict, tpr: Dict, roc_auc: Dict) -> go.Figure:
        """
        Create ROC curves for all classes
        
        Args:
            fpr: False positive rates per class
            tpr: True positive rates per class
            roc_auc: AUC scores per class
            
        Returns:
            Plotly figure
        """
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        fig = go.Figure()
        
        # Plot ROC for each class
        for i, (label, color) in enumerate(zip(self.class_labels, colors)):
            fig.add_trace(go.Scatter(
                x=fpr[i], y=tpr[i],
                name=f'{label} (AUC = {roc_auc[i]:.3f})',
                mode='lines',
                line=dict(color=color, width=2)
            ))
        
        # Add micro-average
        fig.add_trace(go.Scatter(
            x=fpr['micro'], y=tpr['micro'],
            name=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
            mode='lines',
            line=dict(color='#9b59b6', dash='dash', width=2)
        ))
        
        # Add diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Classifier',
            mode='lines',
            line=dict(color='gray', dash='dot')
        ))
        
        fig.update_layout(
            title={
                'text': '📈 ROC Curves (Multi-Class)',
                'font': {'size': 20}
            },
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            width=700,
            legend=dict(x=0.6, y=0.1),
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05])
        )
        
        return fig
    
    def plot_metrics_comparison(self, metrics: Dict) -> go.Figure:
        """
        Create bar chart comparing precision, recall, F1 per class
        
        Args:
            metrics: Dictionary with evaluation metrics
            
        Returns:
            Plotly figure
        """
        precision = metrics['precision_per_class']
        recall = metrics['recall_per_class']
        f1 = metrics['f1_per_class']
        
        fig = go.Figure()
        
        x = self.class_labels
        
        fig.add_trace(go.Bar(
            name='Precision',
            x=x, y=precision,
            marker_color='#3498db',
            text=[f'{v:.2%}' for v in precision],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Recall',
            x=x, y=recall,
            marker_color='#e74c3c',
            text=[f'{v:.2%}' for v in recall],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='F1-Score',
            x=x, y=f1,
            marker_color='#2ecc71',
            text=[f'{v:.2%}' for v in f1],
            textposition='outside'
        ))
        
        fig.update_layout(
            title={
                'text': '📊 Per-Class Performance Metrics',
                'font': {'size': 20}
            },
            xaxis_title='Risk Category',
            yaxis_title='Score',
            barmode='group',
            height=450,
            width=700,
            yaxis=dict(range=[0, 1.15]),
            legend=dict(x=0.8, y=1.0)
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict) -> go.Figure:
        """
        Create horizontal bar chart for feature importance
        
        Args:
            feature_importance: Dictionary mapping feature names to importance
            
        Returns:
            Plotly figure
        """
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importances)
        features = [features[i] for i in sorted_idx]
        importances = [importances[i] for i in sorted_idx]
        
        # Color gradient
        colors = px.colors.sequential.Blues[2:]
        n_colors = len(colors)
        bar_colors = [colors[min(int(i * n_colors / len(features)), n_colors-1)] 
                      for i in range(len(features))]
        
        fig = go.Figure(go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker_color=bar_colors,
            text=[f'{v:.3f}' for v in importances],
            textposition='outside'
        ))
        
        fig.update_layout(
            title={
                'text': '🎯 Feature Importance (AdaBoost)',
                'font': {'size': 20}
            },
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=400,
            width=600,
            margin=dict(l=150)
        )
        
        return fig
    
    def plot_accuracy_gauge(self, accuracy: float) -> go.Figure:
        """
        Create gauge chart for accuracy
        
        Args:
            accuracy: Accuracy score (0-1)
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=accuracy * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Model Accuracy", 'font': {'size': 24}},
            number={'suffix': '%', 'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#2ecc71"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#e74c3c'},
                    {'range': [50, 70], 'color': '#f39c12'},
                    {'range': [70, 85], 'color': '#3498db'},
                    {'range': [85, 100], 'color': '#2ecc71'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': accuracy * 100
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            width=400,
            margin=dict(t=50, b=0)
        )
        
        return fig
    
    def get_metrics_summary(self, metrics: Dict) -> pd.DataFrame:
        """
        Create summary DataFrame of all metrics
        
        Args:
            metrics: Dictionary with evaluation metrics
            
        Returns:
            DataFrame with metrics summary
        """
        summary_data = {
            'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 
                       'F1-Score (Macro)', 'AUC (Macro)'],
            'Score': [
                metrics['accuracy'],
                metrics['precision_macro'],
                metrics['recall_macro'],
                metrics['f1_macro'],
                metrics.get('auc_macro', 'N/A')
            ],
            'Percentage': [
                f"{metrics['accuracy']:.2%}",
                f"{metrics['precision_macro']:.2%}",
                f"{metrics['recall_macro']:.2%}",
                f"{metrics['f1_macro']:.2%}",
                f"{metrics.get('auc_macro', 0):.2%}" if metrics.get('auc_macro') else 'N/A'
            ]
        }
        
        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Quick test
    from adaboost_model import TaxEvasionAdaBoost
    from training_data import generate_training_data, get_feature_columns
    
    print("Testing ModelEvaluator...")
    
    # Generate data and train model
    df = generate_training_data(500)
    model = TaxEvasionAdaBoost()
    X = df[get_feature_columns()]
    y = df['risk_label'].values
    
    results = model.train(X, y)
    
    # Get predictions
    y_pred = model.model.predict(results['X_test'])
    y_proba = model.model.predict_proba(results['X_test'])
    y_test = results['y_test']
    
    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_proba)
    
    print("\n✅ Evaluation Complete!")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Precision (macro): {metrics['precision_macro']:.2%}")
    print(f"Recall (macro): {metrics['recall_macro']:.2%}")
    print(f"F1-Score (macro): {metrics['f1_macro']:.2%}")
    print(f"AUC (macro): {metrics.get('auc_macro', 'N/A')}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
