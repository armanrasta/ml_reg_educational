import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

class ModelEvaluator:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
    def evaluate_model(self, y_true, y_pred, model_name):
        metrics = {
            'R2 Score': r2_score(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred)
        }
        
        # Save metrics
        self._save_metrics(metrics, model_name)
        
        # Create visualizations
        self._create_prediction_plot(y_true, y_pred, model_name)
        
        return metrics
    
    def _save_metrics(self, metrics, model_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/{model_name}_metrics_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def _create_prediction_plot(self, y_true, y_pred, model_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers',
                                name='Predictions'))
        fig.add_trace(go.Scatter(x=[y_true.min(), y_true.max()],
                                y=[y_true.min(), y_true.max()],
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        
        fig.update_layout(
            title=f'{model_name} - Actual vs Predicted Values',
            xaxis_title='Actual Price',
            yaxis_title='Predicted Price',
            template='plotly_dark'
        )
        
        fig.write_html(f"{self.results_dir}/{model_name}_predictions_{timestamp}.html")

def generate_report(model_results, feature_importance, dataset_info):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = f"""
    # Car Price Prediction Model Report
    Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    ## Dataset Information
    - Total samples: {dataset_info['total_samples']}
    - Features used: {dataset_info['features']}
    - Training samples: {dataset_info['train_samples']}
    - Test samples: {dataset_info['test_samples']}

    ## Model Performance Summary
    """
    
    for model_name, metrics in model_results.items():
        report += f"\n### {model_name}\n"
        for metric, value in metrics.items():
            report += f"- {metric}: {value:.4f}\n"
    
    report += "\n## Feature Importance\n"
    for feature, importance in feature_importance.items():
        report += f"- {feature}: {importance:.4f}\n"
    
    with open(f"results/model_report_{timestamp}.md", 'w') as f:
        f.write(report)
