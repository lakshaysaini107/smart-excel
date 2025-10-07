import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalyticsEngine:
    """Advanced analytics capabilities for Excel data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        
    def perform_trend_analysis(self, data: pd.DataFrame, 
                             date_column: str, value_column: str) -> Dict:
        """
        Comprehensive trend analysis with seasonal decomposition
        """
        try:
            # Prepare time series data
            df = data.copy()
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
            df.set_index(date_column, inplace=True)
            
            time_series = df[value_column].fillna(method='ffill')
            
            results = {
                'trend_direction': self._calculate_trend_direction(time_series),
                'trend_strength': self._calculate_trend_strength(time_series),
                'seasonal_patterns': {},
                'anomalies': self._detect_anomalies(time_series),
                'statistical_summary': time_series.describe().to_dict(),
                'growth_rate': self._calculate_growth_rate(time_series),
                'recommendations': []
            }
            
            # Seasonal decomposition (if enough data points)
            if len(time_series) > 24:  # Need sufficient data for seasonality
                try:
                    decomposition = seasonal_decompose(
                        time_series, 
                        model='additive', 
                        period=min(12, len(time_series)//2)
                    )
                    
                    results['seasonal_patterns'] = {
                        'seasonal_strength': np.std(decomposition.seasonal) / np.std(time_series),
                        'trend_component': decomposition.trend.dropna().tolist(),
                        'seasonal_component': decomposition.seasonal.dropna().tolist(),
                        'residual_component': decomposition.resid.dropna().tolist()
                    }
                except Exception as e:
                    logger.warning(f"Seasonal decomposition failed: {e}")
            
            # Generate recommendations
            results['recommendations'] = self._generate_trend_recommendations(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {'error': str(e)}
    
    def perform_predictive_analysis(self, data: pd.DataFrame, 
                                   target_column: str, 
                                   feature_columns: List[str] = None,
                                   prediction_horizon: int = 5) -> Dict:
        """
        Predictive analysis using multiple algorithms
        """
        try:
            df = data.copy()
            
            # Prepare features and target
            if feature_columns is None:
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                feature_columns = [col for col in numeric_columns if col != target_column]
            
            X = df[feature_columns].fillna(df[feature_columns].mean())
            y = df[target_column].fillna(df[target_column].mean())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train multiple models
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression()
            }
            
            results = {
                'model_performance': {},
                'feature_importance': {},
                'predictions': {},
                'recommendations': []
            }
            
            best_model = None
            best_score = -float('inf')
            
            for model_name, model in models.items():
                # Train model
                if model_name == 'Random Forest':
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    feature_imp = dict(zip(feature_columns, model.feature_importances_))
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    feature_imp = dict(zip(feature_columns, abs(model.coef_)))
                
                # Evaluate model
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results['model_performance'][model_name] = {
                    'mean_absolute_error': mae,
                    'r2_score': r2,
                    'accuracy_percentage': max(0, r2 * 100)
                }
                
                results['feature_importance'][model_name] = feature_imp
                
                # Select best model
                if r2 > best_score:
                    best_score = r2
                    best_model = (model_name, model)
            
            # Generate predictions with best model
            if best_model:
                model_name, model = best_model
                
                # Future predictions (simplified approach)
                if len(df) > prediction_horizon:
                    recent_data = X.tail(prediction_horizon)
                    if model_name == 'Random Forest':
                        predictions = model.predict(recent_data)
                    else:
                        predictions = model.predict(self.scaler.transform(recent_data))
                    
                    results['predictions'] = {
                        'best_model': model_name,
                        'predicted_values': predictions.tolist(),
                        'confidence_interval': self._calculate_prediction_intervals(predictions)
                    }
            
            # Generate recommendations
            results['recommendations'] = self._generate_prediction_recommendations(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Predictive analysis failed: {e}")
            return {'error': str(e)}
    
    def perform_comparative_analysis(self, data: pd.DataFrame, 
                                   group_column: str, 
                                   value_column: str) -> Dict:
        """
        Comparative analysis between groups
        """
        try:
            results = {
                'group_statistics': {},
                'statistical_tests': {},
                'rankings': {},
                'recommendations': []
            }
            
            # Group statistics
            grouped = data.groupby(group_column)[value_column]
            
            for group_name, group_data in grouped:
                results['group_statistics'][str(group_name)] = {
                    'mean': group_data.mean(),
                    'median': group_data.median(),
                    'std': group_data.std(),
                    'min': group_data.min(),
                    'max': group_data.max(),
                    'count': len(group_data)
                }
            
            # Statistical significance tests
            groups = [group.dropna() for name, group in grouped]
            
            if len(groups) >= 2:
                # ANOVA test for multiple groups
                f_stat, p_value = stats.f_oneway(*groups)
                results['statistical_tests']['anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                # Pairwise t-tests for groups
                if len(groups) == 2:
                    t_stat, t_p_value = stats.ttest_ind(groups, groups)
                    results['statistical_tests']['t_test'] = {
                        't_statistic': t_stat,
                        'p_value': t_p_value,
                        'significant': t_p_value < 0.05
                    }
            
            # Rankings
            group_means = {name: stats['mean'] for name, stats in results['group_statistics'].items()}
            sorted_groups = sorted(group_means.items(), key=lambda x: x, reverse=True)
            results['rankings'] = {
                'by_average': [{'group': group, 'value': value} for group, value in sorted_groups]
            }
            
            # Recommendations
            results['recommendations'] = self._generate_comparison_recommendations(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_trend_direction(self, series: pd.Series) -> str:
        """Calculate overall trend direction"""
        if len(series) < 2:
            return 'insufficient_data'
        
        # Linear regression to find trend
        x = np.arange(len(series))
        slope, _, r_value, _, _ = stats.linregress(x, series)
        
        if abs(r_value) < 0.3:
            return 'no_clear_trend'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _calculate_trend_strength(self, series: pd.Series) -> float:
        """Calculate trend strength (0-1 scale)"""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        _, _, r_value, _, _ = stats.linregress(x, series)
        return abs(r_value)  # R-squared indicates trend strength
    
    def _detect_anomalies(self, series: pd.Series) -> List[Dict]:
        """Detect anomalies in time series data"""
        try:
            # Use IQR method for anomaly detection
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomalies = []
            for idx, value in series.items():
                if value < lower_bound or value > upper_bound:
                    anomalies.append({
                        'timestamp': str(idx),
                        'value': float(value),
                        'type': 'high' if value > upper_bound else 'low',
                        'deviation': abs(value - series.median())
                    })
            
            return anomalies
            
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return []
    
    def _calculate_growth_rate(self, series: pd.Series) -> Dict:
        """Calculate various growth rates"""
        try:
            # Overall growth rate
            if len(series) < 2:
                return {'error': 'insufficient_data'}
            
            first_value = series.iloc
            last_value = series.iloc[-1]
            
            if first_value == 0:
                overall_growth = float('inf') if last_value > 0 else 0
            else:
                overall_growth = ((last_value - first_value) / first_value) * 100
            
            # Period-over-period growth rates
            pct_changes = series.pct_change().dropna() * 100
            
            return {
                'overall_growth_rate': overall_growth,
                'average_period_growth': pct_changes.mean(),
                'growth_volatility': pct_changes.std(),
                'periods_with_growth': (pct_changes > 0).sum(),
                'periods_with_decline': (pct_changes < 0).sum()
            }
            
        except Exception as e:
            logger.warning(f"Growth rate calculation failed: {e}")
            return {'error': str(e)}
    
    def _generate_trend_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on trend analysis results"""
        recommendations = []
        
        try:
            trend_direction = results.get('trend_direction', 'unknown')
            trend_strength = results.get('trend_strength', 0)
            growth_rate = results.get('growth_rate', {})
            
            if trend_direction == 'increasing':
                if trend_strength > 0.7:
                    recommendations.append("Strong upward trend detected - consider capitalizing on growth momentum")
                else:
                    recommendations.append("Moderate upward trend - monitor for sustainability")
            elif trend_direction == 'decreasing':
                if trend_strength > 0.7:
                    recommendations.append("Strong downward trend - investigate causes and consider corrective actions")
                else:
                    recommendations.append("Moderate decline - monitor closely for improvement opportunities")
            else:
                recommendations.append("No clear trend detected - data may be stable or volatile")
            
            # Growth rate recommendations
            if growth_rate.get('overall_growth_rate', 0) > 20:
                recommendations.append("High growth rate observed - ensure sustainable scaling strategies")
            elif growth_rate.get('overall_growth_rate', 0) < -20:
                recommendations.append("Significant decline detected - urgent action may be required")
            
            # Seasonal pattern recommendations
            seasonal_patterns = results.get('seasonal_patterns', {})
            if seasonal_patterns.get('seasonal_strength', 0) > 0.3:
                recommendations.append("Strong seasonal patterns detected - consider seasonal adjustments in planning")
            
            # Anomaly recommendations
            anomalies = results.get('anomalies', [])
            if len(anomalies) > 0:
                recommendations.append(f"Found {len(anomalies)} anomalies - investigate for data quality or business insights")
            
        except Exception as e:
            logger.warning(f"Error generating trend recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations")
        
        return recommendations
    
    def _generate_prediction_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on predictive analysis results"""
        recommendations = []
        
        try:
            model_performance = results.get('model_performance', {})
            predictions = results.get('predictions', {})
            
            # Model performance recommendations
            if model_performance:
                best_model = predictions.get('best_model', 'Unknown')
                best_score = max([perf.get('r2_score', 0) for perf in model_performance.values()], default=0)
                
                if best_score > 0.8:
                    recommendations.append(f"Excellent model performance ({best_model}) - predictions are highly reliable")
                elif best_score > 0.6:
                    recommendations.append(f"Good model performance ({best_model}) - predictions are reasonably reliable")
                elif best_score > 0.4:
                    recommendations.append(f"Moderate model performance ({best_model}) - use predictions with caution")
                else:
                    recommendations.append(f"Poor model performance ({best_model}) - consider improving data quality or model selection")
            
            # Feature importance recommendations
            feature_importance = results.get('feature_importance', {})
            if feature_importance:
                for model_name, features in feature_importance.items():
                    if features:
                        top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:3]
                        recommendations.append(f"Top predictive features for {model_name}: {', '.join([f[0] for f in top_features])}")
            
            # Prediction confidence recommendations
            confidence_interval = predictions.get('confidence_interval', {})
            if confidence_interval:
                recommendations.append("Confidence intervals available - consider uncertainty in decision making")
            
        except Exception as e:
            logger.warning(f"Error generating prediction recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations")
        
        return recommendations
    
    def _generate_comparison_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on comparative analysis results"""
        recommendations = []
        
        try:
            group_statistics = results.get('group_statistics', {})
            statistical_tests = results.get('statistical_tests', {})
            rankings = results.get('rankings', {})
            
            # Statistical significance recommendations
            anova = statistical_tests.get('anova', {})
            if anova.get('significant', False):
                recommendations.append("Statistically significant differences found between groups - investigate causes")
            else:
                recommendations.append("No significant differences between groups - groups may be similar")
            
            # Ranking recommendations
            if rankings.get('by_average'):
                top_performers = rankings['by_average'][:3]
                recommendations.append(f"Top performing groups: {', '.join([str(item['group']) for item in top_performers])}")
                
                # Performance gap analysis
                if len(rankings['by_average']) > 1:
                    top_value = rankings['by_average'][0]['value']
                    bottom_value = rankings['by_average'][-1]['value']
                    gap = top_value - bottom_value
                    if gap > top_value * 0.5:  # 50% gap
                        recommendations.append("Large performance gap detected - investigate best practices from top performers")
            
            # Group size recommendations
            for group_name, stats in group_statistics.items():
                count = stats.get('count', 0)
                if count < 10:
                    recommendations.append(f"Group '{group_name}' has small sample size ({count}) - results may not be reliable")
            
        except Exception as e:
            logger.warning(f"Error generating comparison recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations")
        
        return recommendations
    
    def _calculate_prediction_intervals(self, predictions: np.ndarray) -> Dict:
        """Calculate prediction confidence intervals"""
        try:
            if len(predictions) == 0:
                return {'lower': [], 'upper': [], 'mean': []}
            
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            # Simple confidence interval (95%)
            margin_error = 1.96 * std_pred
            
            return {
                'lower': (mean_pred - margin_error).tolist() if isinstance(mean_pred, np.ndarray) else [mean_pred - margin_error],
                'upper': (mean_pred + margin_error).tolist() if isinstance(mean_pred, np.ndarray) else [mean_pred + margin_error],
                'mean': mean_pred.tolist() if isinstance(mean_pred, np.ndarray) else [mean_pred]
            }
        except Exception as e:
            logger.warning(f"Error calculating prediction intervals: {e}")
            return {'lower': [], 'upper': [], 'mean': []}