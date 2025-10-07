import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AdvancedVisualizationEngine:
    """Advanced visualization capabilities with Plotly"""
    
    def __init__(self):
        # Set default theme
        self.default_theme = 'plotly_white'
        self.color_palette = px.colors.qualitative.Set3
    
    def create_trend_visualization(self, data: pd.DataFrame, 
                                 x_column: str, y_column: str,
                                 title: str = "Trend Analysis") -> Dict:
        """Create comprehensive trend visualization"""
        try:
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Main Trend', 'Distribution', 'Moving Average', 'Growth Rate'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": True}, {"secondary_y": False}]]
            )
            
            # Main trend line
            fig.add_trace(
                go.Scatter(
                    x=data[x_column], y=data[y_column],
                    mode='lines+markers',
                    name='Actual Values',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Distribution (histogram)
            fig.add_trace(
                go.Histogram(
                    x=data[y_column],
                    name='Distribution',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
            
            # Moving average
            if len(data) > 7:
                window_size = min(7, len(data) // 4)
                moving_avg = data[y_column].rolling(window=window_size).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=data[x_column], y=data[y_column],
                        mode='lines', name='Original',
                        line=dict(color='lightgray', width=1)
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data[x_column], y=moving_avg,
                        mode='lines', name=f'{window_size}-period MA',
                        line=dict(color='red', width=2)
                    ),
                    row=2, col=1
                )
            
            # Growth rate
            growth_rate = data[y_column].pct_change() * 100
            fig.add_trace(
                go.Bar(
                    x=data[x_column][1:], y=growth_rate.dropna(),
                    name='Growth Rate (%)',
                    marker_color='green'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text=title,
                showlegend=True,
                template=self.default_theme,
                height=600
            )
            
            return {
                'figure': fig,
                'type': 'trend_analysis',
                'interactive': True
            }
            
        except Exception as e:
            logger.error(f"Trend visualization failed: {e}")
            return {'error': str(e)}
    
    def create_comparison_visualization(self, data: pd.DataFrame,
                                     group_column: str, value_column: str,
                                     title: str = "Comparative Analysis") -> Dict:
        """Create comprehensive comparison visualization"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Box Plot', 'Bar Chart', 'Violin Plot', 'Statistics Table'),
                specs=[[{"type": "box"}, {"type": "bar"}],
                       [{"type": "violin"}, {"type": "table"}]]
            )
            
            # Box plot
            for i, group in enumerate(data[group_column].unique()):
                group_data = data[data[group_column] == group][value_column]
                fig.add_trace(
                    go.Box(
                        y=group_data,
                        name=str(group),
                        marker_color=self.color_palette[i % len(self.color_palette)]
                    ),
                    row=1, col=1
                )
            
            # Bar chart with averages
            group_means = data.groupby(group_column)[value_column].mean()
            fig.add_trace(
                go.Bar(
                    x=group_means.index.astype(str),
                    y=group_means.values,
                    name='Average Values',
                    marker_color=self.color_palette[:len(group_means)]
                ),
                row=1, col=2
            )
            
            # Violin plot
            for i, group in enumerate(data[group_column].unique()):
                group_data = data[data[group_column] == group][value_column]
                fig.add_trace(
                    go.Violin(
                        y=group_data,
                        name=str(group),
                        marker_color=self.color_palette[i % len(self.color_palette)]
                    ),
                    row=2, col=1
                )
            
            # Statistics table
            stats_data = []
            for group in data[group_column].unique():
                group_data = data[data[group_column] == group][value_column]
                stats_data.append([
                    str(group),
                    f"{group_data.mean():.2f}",
                    f"{group_data.std():.2f}",
                    f"{group_data.min():.2f}",
                    f"{group_data.max():.2f}",
                    f"{len(group_data)}"
                ])
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Group', 'Mean', 'Std', 'Min', 'Max', 'Count']),
                    cells=dict(values=list(zip(*stats_data)))
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text=title,
                showlegend=True,
                template=self.default_theme,
                height=700
            )
            
            return {
                'figure': fig,
                'type': 'comparison_analysis',
                'interactive': True
            }
            
        except Exception as e:
            logger.error(f"Comparison visualization failed: {e}")
            return {'error': str(e)}
    
    def create_predictive_visualization(self, historical_data: pd.DataFrame,
                                      predictions: List[float],
                                      x_column: str, y_column: str,
                                      title: str = "Predictive Analysis") -> Dict:
        """Create predictive analysis visualization"""
        try:
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=historical_data[x_column],
                    y=historical_data[y_column],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Predictions
            if predictions:
                # Create future dates/indices
                last_x = historical_data[x_column].iloc[-1]
                
                if pd.api.types.is_datetime64_any_dtype(historical_data[x_column]):
                    # For datetime index
                    freq = pd.infer_freq(historical_data[x_column])
                    future_dates = pd.date_range(
                        start=last_x, 
                        periods=len(predictions) + 1, 
                        freq=freq or 'D'
                    )[1:]
                else:
                    # For numeric index
                    step = historical_data[x_column].iloc[-1] - historical_data[x_column].iloc[-2]
                    future_dates = [last_x + (i + 1) * step for i in range(len(predictions))]
                
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=predictions,
                        mode='lines+markers',
                        name='Predictions',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(symbol='diamond', size=8)
                    )
                )
                
                # Add confidence intervals if available
                # (Simplified - in practice you'd calculate actual confidence intervals)
                std_dev = np.std(historical_data[y_column])
                upper_bound = [p + 1.96 * std_dev for p in predictions]
                lower_bound = [p - 1.96 * std_dev for p in predictions]
                
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=upper_bound,
                        mode='lines',
                        name='Upper Confidence',
                        line=dict(color='rgba(255,0,0,0.3)'),
                        showlegend=False
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=lower_bound,
                        mode='lines',
                        name='Confidence Interval',
                        line=dict(color='rgba(255,0,0,0.3)'),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.1)'
                    )
                )
            
            fig.update_layout(
                title=title,
                xaxis_title=x_column,
                yaxis_title=y_column,
                template=self.default_theme,
                hovermode='x unified'
            )
            
            return {
                'figure': fig,
                'type': 'predictive_analysis',
                'interactive': True
            }
            
        except Exception as e:
            logger.error(f"Predictive visualization failed: {e}")
            return {'error': str(e)}
    
    def create_correlation_heatmap(self, data: pd.DataFrame,
                                 title: str = "Correlation Analysis") -> Dict:
        """Create correlation heatmap for numerical columns"""
        try:
            # Select only numerical columns
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return {'error': 'No numerical data available for correlation analysis'}
            
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=title,
                template=self.default_theme,
                height=500,
                width=500
            )
            
            return {
                'figure': fig,
                'type': 'correlation_heatmap',
                'interactive': True,
                'correlation_matrix': corr_matrix.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Correlation heatmap failed: {e}")
            return {'error': str(e)}
    
    def create_dashboard_layout(self, figures: List[go.Figure],
                              titles: List[str] = None) -> Dict:
        """Create dashboard with multiple visualizations"""
        try:
            num_figures = len(figures)
            
            if num_figures == 0:
                return {'error': 'No figures provided'}
            
            # Determine layout
            if num_figures <= 2:
                rows, cols = 1, num_figures
            elif num_figures <= 4:
                rows, cols = 2, 2
            else:
                rows = int(np.ceil(np.sqrt(num_figures)))
                cols = int(np.ceil(num_figures / rows))
            
            # Create subplots
            subplot_titles = titles or [f"Chart {i+1}" for i in range(num_figures)]
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=subplot_titles[:num_figures],
                vertical_spacing=0.08,
                horizontal_spacing=0.08
            )
            
            # Add figures to subplots
            for i, source_fig in enumerate(figures):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                # Add all traces from source figure
                for trace in source_fig.data:
                    fig.add_trace(trace, row=row, col=col)
            
            fig.update_layout(
                title_text="Analytics Dashboard",
                showlegend=True,
                template=self.default_theme,
                height=300 * rows
            )
            
            return {
                'figure': fig,
                'type': 'dashboard',
                'interactive': True,
                'layout': f"{rows}x{cols}"
            }
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            return {'error': str(e)}
