import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger(__name__)

class ResultVisualizer:
    """
    Advanced visualization for model results and predictions.
    Supports static matplotlib plots and interactive Plotly visualizations.
    """
    
    def __init__(self, style: str = 'seaborn'):
        self.style = style
        self.set_plot_style(style)
    
    def set_plot_style(self, style: str) -> None:
        """Set the plotting style."""
        if style == 'seaborn':
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        elif style == 'matplotlib':
            plt.style.use('default')
        elif style == 'dark':
            plt.style.use('dark_background')
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        engine_id: Optional[int] = None, 
                        interactive: bool = False) -> Any:
        """
        Plot true vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            engine_id: Optional engine ID for title
            interactive: Whether to use Plotly
            
        Returns:
            Plot object
        """
        if interactive:
            return self._plot_predictions_interactive(y_true, y_pred, engine_id)
        else:
            return self._plot_predictions_static(y_true, y_pred, engine_id)
    
    def _plot_predictions_static(self, y_true: np.ndarray, y_pred: np.ndarray,
                               engine_id: Optional[int] = None) -> plt.Figure:
        """Static matplotlib version."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.6, s=20)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('True RUL')
        ax1.set_ylabel('Predicted RUL')
        ax1.set_title('True vs Predicted RUL')
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted RUL')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals Plot')
        ax2.grid(True, alpha=0.3)
        
        title = f'RUL Predictions - Engine {engine_id}' if engine_id else 'RUL Predictions'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _plot_predictions_interactive(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    engine_id: Optional[int] = None) -> go.Figure:
        """Interactive Plotly version."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('True vs Predicted RUL', 'Residuals Plot'),
            horizontal_spacing=0.1
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=y_true, y=y_pred, mode='markers',
                marker=dict(size=6, opacity=0.6, color='blue'),
                name='Predictions'
            ),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', line=dict(dash='dash', color='red'),
                name='Perfect Prediction'
            ),
            row=1, col=1
        )
        
        # Residuals plot
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals, mode='markers',
                marker=dict(size=6, opacity=0.6, color='green'),
                name='Residuals'
            ),
            row=1, col=2
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        title = f'RUL Predictions - Engine {engine_id}' if engine_id else 'RUL Predictions'
        fig.update_layout(
            title=title,
            showlegend=True,
            width=1000,
            height=500
        )
        
        fig.update_xaxes(title_text='True RUL', row=1, col=1)
        fig.update_yaxes(title_text='Predicted RUL', row=1, col=1)
        fig.update_xaxes(title_text='Predicted RUL', row=1, col=2)
        fig.update_yaxes(title_text='Residuals', row=1, col=2)
        
        return fig
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            interactive: bool = False) -> Any:
        """
        Plot training history (loss and metrics).
        
        Args:
            history: Training history dictionary
            interactive: Whether to use Plotly
            
        Returns:
            Plot object
        """
        if interactive:
            return self._plot_training_history_interactive(history)
        else:
            return self._plot_training_history_static(history)
    
    def _plot_training_history_static(self, history: Dict[str, List[float]]) -> plt.Figure:
        """Static training history plot."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metrics plot (excluding loss)
        metrics = [m for m in history.keys() if m != 'loss' and m != 'val_loss']
        for metric in metrics:
            if 'val_' in metric:
                continue
            if metric in history:
                axes[1].plot(history[metric], label=f'Training {metric}', linewidth=2)
            if f'val_{metric}' in history:
                axes[1].plot(history[f'val_{metric}'], label=f'Validation {metric}', linewidth=2)
        
        if metrics:
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Metric Value')
            axes[1].set_title('Training and Validation Metrics')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_training_history_interactive(self, history: Dict[str, List[float]]) -> go.Figure:
        """Interactive training history plot."""
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Metrics'))
        
        # Loss traces
        if 'loss' in history:
            fig.add_trace(
                go.Scatter(y=history['loss'], mode='lines', name='Training Loss'),
                row=1, col=1
            )
        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(y=history['val_loss'], mode='lines', name='Validation Loss'),
                row=1, col=1
            )
        
        # Metric traces
        metrics = [m for m in history.keys() if m != 'loss' and m != 'val_loss']
        for i, metric in enumerate(metrics):
            if 'val_' in metric:
                continue
            
            color = px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
            
            if metric in history:
                fig.add_trace(
                    go.Scatter(y=history[metric], mode='lines', 
                              name=f'Training {metric}', line=dict(color=color)),
                    row=1, col=2
                )
            if f'val_{metric}' in history:
                fig.add_trace(
                    go.Scatter(y=history[f'val_{metric}'], mode='lines',
                              name=f'Validation {metric}', line=dict(color=color, dash='dash')),
                    row=1, col=2
                )
        
        fig.update_layout(
            title='Training History',
            showlegend=True,
            width=1000,
            height=500
        )
        
        fig.update_xaxes(title_text='Epoch', row=1, col=1)
        fig.update_yaxes(title_text='Loss', row=1, col=1)
        fig.update_xaxes(title_text='Epoch', row=1, col=2)
        fig.update_yaxes(title_text='Metric Value', row=1, col=2)
        
        return fig
    
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                              interactive: bool = False) -> Any:
        """
        Plot error distribution.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            interactive: Whether to use Plotly
            
        Returns:
            Plot object
        """
        errors = y_true - y_pred
        
        if interactive:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=errors, nbinsx=50, name='Errors'))
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(
                title='Error Distribution',
                xaxis_title='Error (True - Predicted)',
                yaxis_title='Frequency'
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Error (True - Predicted)')
            ax.set_ylabel('Frequency')
            ax.set_title('Error Distribution')
            ax.grid(True, alpha=0.3)
            return fig
    
    def plot_engine_degradation(self, engine_data: pd.DataFrame, 
                              predictions: Optional[pd.DataFrame] = None,
                              engine_id: int = None,
                              interactive: bool = False) -> Any:
        """
        Plot engine degradation over time.
        
        Args:
            engine_data: Engine data with RUL
            predictions: Optional predictions DataFrame
            engine_id: Engine ID for title
            interactive: Whether to use Plotly
            
        Returns:
            Plot object
        """
        if interactive:
            return self._plot_engine_degradation_interactive(engine_data, predictions, engine_id)
        else:
            return self._plot_engine_degradation_static(engine_data, predictions, engine_id)
    
    def _plot_engine_degradation_static(self, engine_data: pd.DataFrame,
                                      predictions: Optional[pd.DataFrame],
                                      engine_id: int) -> plt.Figure:
        """Static engine degradation plot."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # RUL plot
        cycles = engine_data['Cycle']
        true_rul = engine_data['RUL']
        
        ax1.plot(cycles, true_rul, 'b-', linewidth=2, label='True RUL')
        
        if predictions is not None:
            pred_cycles = predictions['Cycle']
            pred_rul = predictions['Predicted_RUL']
            ax1.plot(pred_cycles, pred_rul, 'r--', linewidth=2, label='Predicted RUL')
        
        ax1.set_xlabel('Cycle')
        ax1.set_ylabel('RUL')
        ax1.set_title(f'RUL Degradation - Engine {engine_id}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # RUL decreases over time
        
        # Sensor data plot (example sensors)
        sensor_cols = [col for col in engine_data.columns if 'Sensor' in col]
        if len(sensor_cols) > 0:
            for i, sensor in enumerate(sensor_cols[:3]):  # Plot first 3 sensors
                ax2.plot(cycles, engine_data[sensor], 
                        label=sensor, linewidth=1, alpha=0.7)
            ax2.set_xlabel('Cycle')
            ax2.set_ylabel('Sensor Value')
            ax2.set_title('Sensor Measurements')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_engine_degradation_interactive(self, engine_data: pd.DataFrame,
                                           predictions: Optional[pd.DataFrame],
                                           engine_id: int) -> go.Figure:
        """Interactive engine degradation plot."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'RUL Degradation - Engine {engine_id}', 'Sensor Measurements'),
            vertical_spacing=0.1
        )
        
        cycles = engine_data['Cycle']
        true_rul = engine_data['RUL']
        
        # True RUL
        fig.add_trace(
            go.Scatter(x=cycles, y=true_rul, mode='lines', 
                      name='True RUL', line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        # Predicted RUL
        if predictions is not None:
            pred_cycles = predictions['Cycle']
            pred_rul = predictions['Predicted_RUL']
            fig.add_trace(
                go.Scatter(x=pred_cycles, y=pred_rul, mode='lines',
                          name='Predicted RUL', line=dict(color='red', width=3, dash='dash')),
                row=1, col=1
            )
        
        # Sensor data
        sensor_cols = [col for col in engine_data.columns if 'Sensor' in col]
        colors = px.colors.qualitative.Set1
        for i, sensor in enumerate(sensor_cols[:3]):
            fig.add_trace(
                go.Scatter(x=cycles, y=engine_data[sensor], mode='lines',
                          name=sensor, line=dict(color=colors[i % len(colors)], width=2)),
                row=2, col=1
            )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f'Engine {engine_id} Analysis'
        )
        
        fig.update_yaxes(title_text='RUL', row=1, col=1)
        fig.update_xaxes(title_text='Cycle', row=1, col=1)
        fig.update_yaxes(title_text='Sensor Value', row=2, col=1)
        fig.update_xaxes(title_text='Cycle', row=2, col=1)
        
        return fig
    
    def create_dashboard(self, experiment_results: Dict[str, Any]) -> go.Figure:
        """
        Create a comprehensive dashboard of experiment results.
        
        Args:
            experiment_results: Dictionary containing experiment results
            
        Returns:
            Plotly Figure with dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'True vs Predicted RUL',
                'Error Distribution', 
                'Training History',
                'Residuals Analysis'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        
        return fig