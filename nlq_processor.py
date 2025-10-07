# Advanced Natural Language Query Processor

import pandas as pd
import numpy as np
import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class AdvancedNLQProcessor:
    """Advanced Natural Language Query Processor for data analysis"""
    
    def __init__(self):
        # Intent patterns for different types of queries
        self.intent_patterns = {
            'trend_analysis': [
                r'trend|over time|temporal|time series|change|evolve|progress|growth|decline',
                r'week|month|year|quarter|daily|monthly|yearly|quarterly|seasonal'
            ],
            'aggregation': [
                r'total|sum|average|mean|count|maximum|minimum|max|min|aggregate',
                r'how many|how much|total number|overall'
            ],
            'comparison': [
                r'compare|versus|vs|difference|between|against|relative to',
                r'better|worse|higher|lower|more|less|greater|smaller'
            ],
            'correlation': [
                r'correlat|relationship|connect|associate|link|related|depend',
                r'affect|influence|impact|cause|effect'
            ],
            'filtering': [
                r'where|filter|only|specific|particular|select|subset',
                r'equal|greater than|less than|contains|includes'
            ],
            'ranking': [
                r'top|bottom|best|worst|highest|lowest|rank|sort|order',
                r'first|last|leading|trailing'
            ],
            'distribution': [
                r'distribution|spread|variance|range|histogram|frequency',
                r'normal|skewed|outlier|anomaly'
            ]
        }
        
        # Column type detection patterns
        self.column_patterns = {
            'date': r'date|time|created|modified|year|month|day|timestamp',
            'amount': r'price|cost|amount|value|revenue|sales|profit|expense',
            'quantity': r'quantity|count|number|total|sum|volume',
            'category': r'category|type|class|group|segment|department',
            'location': r'location|city|state|country|region|address',
            'name': r'name|title|description|label'
        }
        
        # Initialize TF-IDF for query similarity
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    def process_query(self, query: str, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Main method to process natural language queries
        """
        try:
            logger.info(f"Processing NLQ: {query[:100]}...")
            
            # Step 1: Intent recognition
            intent = self._detect_intent(query)
            
            # Step 2: Entity extraction
            entities = self._extract_entities(query, data)
            
            # Step 3: Query understanding
            query_structure = self._parse_query_structure(query, entities)
            
            # Step 4: Data analysis
            analysis_result = self._perform_analysis(intent, query_structure, entities, data)
            
            # Step 5: Generate response
            response = self._generate_response(query, intent, analysis_result, entities)
            
            return {
                'success': True,
                'query': query,
                'intent': intent,
                'entities': entities,
                'query_structure': query_structure,
                'analysis_result': analysis_result,
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"NLQ processing failed: {e}")
            return {
                'success': False,
                'query': query,
                'error': str(e),
                'response': {
                    'text': f"I'm sorry, I couldn't process your query: {str(e)}",
                    'visualization': None,
                    'data': None
                }
            }
    
    def _detect_intent(self, query: str) -> str:
        """Detect the primary intent of the query"""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            intent_scores[intent] = score
        
        # Return the intent with highest score, or 'general' if no clear intent
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        else:
            return 'general'
    
    def _extract_entities(self, query: str, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Extract entities from the query"""
        entities = {
            'columns': [],
            'values': [],
            'numbers': [],
            'dates': [],
            'sheets': [],
            'operations': []
        }
        
        query_lower = query.lower()
        
        # Extract numbers
        entities['numbers'] = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        
        # Extract potential date references
        date_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b',  # Months
            r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',  # Days
            r'\btoday|yesterday|tomorrow|last week|next week|last month|next month\b'
        ]
        
        for pattern in date_patterns:
            entities['dates'].extend(re.findall(pattern, query_lower))
        
        # Extract column references
        all_columns = []
        for sheet_name, df in data.items():
            entities['sheets'].append(sheet_name)
            all_columns.extend(df.columns.tolist())
        
        # Find column matches in query
        for column in all_columns:
            if column.lower() in query_lower or any(word in column.lower() for word in query_lower.split()):
                entities['columns'].append(column)
        
        # Extract operations
        operation_patterns = {
            'sum': r'sum|total|add up',
            'average': r'average|mean|avg',
            'count': r'count|number of|how many',
            'max': r'maximum|max|highest|largest',
            'min': r'minimum|min|lowest|smallest',
            'group': r'group by|by|per|each'
        }
        
        for op, pattern in operation_patterns.items():
            if re.search(pattern, query_lower):
                entities['operations'].append(op)
        
        return entities
    
    def _parse_query_structure(self, query: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the structure of the query"""
        structure = {
            'target_columns': entities.get('columns', []),
            'operations': entities.get('operations', []),
            'filters': [],
            'groupby': [],
            'orderby': [],
            'limit': None
        }
        
        query_lower = query.lower()
        
        # Extract grouping information
        if 'by' in query_lower and entities['columns']:
            # Simple heuristic: look for categorical columns after "by"
            by_index = query_lower.find(' by ')
            if by_index != -1:
                remaining_query = query_lower[by_index + 4:]
                for col in entities['columns']:
                    if col.lower() in remaining_query:
                        structure['groupby'].append(col)
        
        # Extract top/bottom limits
        top_pattern = r'top\s+(\d+)|first\s+(\d+)'
        bottom_pattern = r'bottom\s+(\d+)|last\s+(\d+)'
        
        top_match = re.search(top_pattern, query_lower)
        bottom_match = re.search(bottom_pattern, query_lower)
        
        if top_match:
            structure['limit'] = int(top_match.group(1) or top_match.group(2))
            structure['orderby'] = 'desc'
        elif bottom_match:
            structure['limit'] = int(bottom_match.group(1) or bottom_match.group(2))
            structure['orderby'] = 'asc'
        
        return structure
    
    def _perform_analysis(self, intent: str, query_structure: Dict, entities: Dict, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform the actual data analysis based on intent and structure"""
        
        # Select the most appropriate sheet
        if len(data) == 1:
            sheet_name = list(data.keys())[0]
            df = data[sheet_name]
        else:
            # Select sheet with most matching columns
            best_sheet = None
            max_matches = 0
            for sheet_name, df in data.items():
                matches = len([col for col in entities['columns'] if col in df.columns])
                if matches > max_matches:
                    max_matches = matches
                    best_sheet = sheet_name
            
            if best_sheet:
                sheet_name = best_sheet
                df = data[sheet_name]
            else:
                sheet_name = list(data.keys())[0]
                df = data[sheet_name]
        
        analysis_result = {
            'sheet_used': sheet_name,
            'data_shape': df.shape,
            'analysis_type': intent,
            'result': None,
            'visualization_data': None
        }
        
        try:
            if intent == 'aggregation':
                analysis_result['result'] = self._perform_aggregation(df, query_structure, entities)
            
            elif intent == 'trend_analysis':
                analysis_result['result'] = self._perform_trend_analysis(df, query_structure, entities)
            
            elif intent == 'comparison':
                analysis_result['result'] = self._perform_comparison(df, query_structure, entities)
            
            elif intent == 'correlation':
                analysis_result['result'] = self._perform_correlation(df, query_structure, entities)
            
            elif intent == 'ranking':
                analysis_result['result'] = self._perform_ranking(df, query_structure, entities)
            
            elif intent == 'distribution':
                analysis_result['result'] = self._perform_distribution(df, query_structure, entities)
            
            else:
                # General analysis - show basic statistics
                analysis_result['result'] = self._perform_general_analysis(df, query_structure, entities)
                
        except Exception as e:
            logger.error(f"Analysis failed for intent {intent}: {e}")
            analysis_result['result'] = {'error': str(e)}
        
        return analysis_result
    
    def _perform_aggregation(self, df: pd.DataFrame, structure: Dict, entities: Dict) -> Dict:
        """Perform aggregation analysis"""
        try:
            result = {'type': 'aggregation', 'data': {}}
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_cols = [col for col in entities['columns'] if col in numeric_cols] or numeric_cols[:3]
            
            if 'sum' in entities['operations']:
                result['data']['sums'] = df[target_cols].sum().to_dict()
            if 'average' in entities['operations'] or 'mean' in entities['operations']:
                result['data']['averages'] = df[target_cols].mean().to_dict()
            if 'count' in entities['operations']:
                result['data']['counts'] = df[target_cols].count().to_dict()
            if 'max' in entities['operations']:
                result['data']['maximums'] = df[target_cols].max().to_dict()
            if 'min' in entities['operations']:
                result['data']['minimums'] = df[target_cols].min().to_dict()
            
            # If no specific operation, provide summary
            if not any(op in entities['operations'] for op in ['sum', 'average', 'count', 'max', 'min']):
                result['data']['summary'] = df[target_cols].describe().to_dict()
            
            # Group by if specified
            if structure['groupby']:
                groupby_col = structure['groupby'][0]
                if groupby_col in df.columns:
                    grouped = df.groupby(groupby_col)[target_cols].agg(['mean', 'sum', 'count'])
                    result['data']['grouped'] = grouped.to_dict()
            
            result['visualization_data'] = self._create_aggregation_viz_data(df, target_cols, structure)
            
            return result
            
        except Exception as e:
            return {'type': 'aggregation', 'error': str(e)}
    
    def _perform_trend_analysis(self, df: pd.DataFrame, structure: Dict, entities: Dict) -> Dict:
        """Perform trend analysis"""
        try:
            result = {'type': 'trend_analysis', 'data': {}}
            
            # Find date columns
            date_cols = []
            for col in df.columns:
                if df[col].dtype in ['datetime64[ns]', 'object']:
                    if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'modified']):
                        date_cols.append(col)
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_cols = [col for col in entities['columns'] if col in numeric_cols] or numeric_cols[:2]
            
            if date_cols and target_cols:
                date_col = date_cols[0]
                value_col = target_cols[0]
                
                # Convert to datetime if needed
                df_temp = df.copy()
                try:
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                    df_temp = df_temp.sort_values(date_col)
                    
                    result['data']['trend'] = {
                        'dates': df_temp[date_col].dt.strftime('%Y-%m-%d').tolist(),
                        'values': df_temp[value_col].tolist(),
                        'column': value_col
                    }
                    
                    # Calculate trend statistics
                    result['data']['trend_stats'] = {
                        'total_change': float(df_temp[value_col].iloc[-1] - df_temp[value_col].iloc[0]),
                        'percent_change': float((df_temp[value_col].iloc[-1] / df_temp[value_col].iloc[0] - 1) * 100),
                        'average': float(df_temp[value_col].mean()),
                        'volatility': float(df_temp[value_col].std())
                    }
                    
                    result['visualization_data'] = self._create_trend_viz_data(df_temp, date_col, value_col)
                    
                except Exception as e:
                    result['data']['error'] = f"Date conversion failed: {str(e)}"
            else:
                result['data']['error'] = "No suitable date and numeric columns found for trend analysis"
            
            return result
            
        except Exception as e:
            return {'type': 'trend_analysis', 'error': str(e)}
    
    def _perform_comparison(self, df: pd.DataFrame, structure: Dict, entities: Dict) -> Dict:
        """Perform comparison analysis"""
        try:
            result = {'type': 'comparison', 'data': {}}
            
            # Get categorical columns for grouping
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            target_cols = [col for col in entities['columns'] if col in numeric_cols] or numeric_cols[:2]
            group_cols = [col for col in entities['columns'] if col in categorical_cols] or categorical_cols[:1]
            
            if group_cols and target_cols:
                group_col = group_cols[0]
                value_col = target_cols[0]
                
                comparison_data = df.groupby(group_col)[value_col].agg(['mean', 'sum', 'count', 'std']).round(2)
                result['data']['comparison'] = comparison_data.to_dict()
                
                # Add ranking
                ranked = comparison_data.sort_values('mean', ascending=False)
                result['data']['ranking'] = {
                    'by_average': ranked.index.tolist(),
                    'values': ranked['mean'].tolist()
                }
                
                result['visualization_data'] = self._create_comparison_viz_data(df, group_col, value_col)
            
            else:
                result['data']['error'] = "No suitable categorical and numeric columns found for comparison"
            
            return result
            
        except Exception as e:
            return {'type': 'comparison', 'error': str(e)}
    
    def _perform_correlation(self, df: pd.DataFrame, structure: Dict, entities: Dict) -> Dict:
        """Perform correlation analysis"""
        try:
            result = {'type': 'correlation', 'data': {}}
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_cols = [col for col in entities['columns'] if col in numeric_cols] or numeric_cols
            
            if len(target_cols) >= 2:
                correlation_matrix = df[target_cols].corr().round(3)
                result['data']['correlation_matrix'] = correlation_matrix.to_dict()
                
                # Find strong correlations
                strong_corr = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_val = correlation_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            strong_corr.append({
                                'var1': correlation_matrix.columns[i],
                                'var2': correlation_matrix.columns[j],
                                'correlation': float(corr_val),
                                'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate'
                            })
                
                result['data']['strong_correlations'] = strong_corr
                result['visualization_data'] = self._create_correlation_viz_data(correlation_matrix)
            
            else:
                result['data']['error'] = "Need at least 2 numeric columns for correlation analysis"
            
            return result
            
        except Exception as e:
            return {'type': 'correlation', 'error': str(e)}
    
    def _perform_ranking(self, df: pd.DataFrame, structure: Dict, entities: Dict) -> Dict:
        """Perform ranking analysis"""
        try:
            result = {'type': 'ranking', 'data': {}}
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_cols = [col for col in entities['columns'] if col in numeric_cols] or numeric_cols[:1]
            
            if target_cols:
                value_col = target_cols[0]
                
                # Sort data
                ascending = structure['orderby'] == 'asc' if structure['orderby'] else False
                sorted_df = df.sort_values(value_col, ascending=ascending)
                
                # Apply limit if specified
                if structure['limit']:
                    sorted_df = sorted_df.head(structure['limit'])
                else:
                    sorted_df = sorted_df.head(10)  # Default top 10
                
                result['data']['ranking'] = {
                    'items': sorted_df.index.tolist(),
                    'values': sorted_df[value_col].tolist(),
                    'column': value_col,
                    'order': 'descending' if not ascending else 'ascending'
                }
                
                # Include other relevant columns
                relevant_cols = [col for col in df.columns if col != value_col][:3]
                for col in relevant_cols:
                    result['data'][col] = sorted_df[col].tolist()
                
                result['visualization_data'] = self._create_ranking_viz_data(sorted_df, value_col)
            
            else:
                result['data']['error'] = "No numeric columns found for ranking"
            
            return result
            
        except Exception as e:
            return {'type': 'ranking', 'error': str(e)}
    
    def _perform_distribution(self, df: pd.DataFrame, structure: Dict, entities: Dict) -> Dict:
        """Perform distribution analysis"""
        try:
            result = {'type': 'distribution', 'data': {}}
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_cols = [col for col in entities['columns'] if col in numeric_cols] or numeric_cols[:1]
            
            if target_cols:
                value_col = target_cols[0]
                
                # Basic distribution statistics
                result['data']['statistics'] = {
                    'mean': float(df[value_col].mean()),
                    'median': float(df[value_col].median()),
                    'std': float(df[value_col].std()),
                    'min': float(df[value_col].min()),
                    'max': float(df[value_col].max()),
                    'q25': float(df[value_col].quantile(0.25)),
                    'q75': float(df[value_col].quantile(0.75))
                }
                
                # Outlier detection
                q1 = df[value_col].quantile(0.25)
                q3 = df[value_col].quantile(0.75)
                iqr = q3 - q1
                outliers = df[(df[value_col] < q1 - 1.5 * iqr) | (df[value_col] > q3 + 1.5 * iqr)]
                
                result['data']['outliers'] = {
                    'count': len(outliers),
                    'percentage': round(len(outliers) / len(df) * 100, 2),
                    'values': outliers[value_col].tolist()[:10]  # First 10 outliers
                }
                
                result['visualization_data'] = self._create_distribution_viz_data(df, value_col)
            
            else:
                result['data']['error'] = "No numeric columns found for distribution analysis"
            
            return result
            
        except Exception as e:
            return {'type': 'distribution', 'error': str(e)}
    
    def _perform_general_analysis(self, df: pd.DataFrame, structure: Dict, entities: Dict) -> Dict:
        """Perform general analysis when intent is unclear"""
        try:
            result = {'type': 'general', 'data': {}}
            
            # Basic data overview
            result['data']['overview'] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict()
            }
            
            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                result['data']['numeric_summary'] = df[numeric_cols].describe().round(3).to_dict()
            
            # Sample data
            result['data']['sample'] = df.head(5).to_dict('records')
            
            result['visualization_data'] = self._create_general_viz_data(df)
            
            return result
            
        except Exception as e:
            return {'type': 'general', 'error': str(e)}
    
    def _create_aggregation_viz_data(self, df: pd.DataFrame, cols: List[str], structure: Dict) -> Dict:
        """Create visualization data for aggregation"""
        if not cols:
            return None
        
        try:
            # Create bar chart for aggregated values
            if structure.get('groupby') and structure['groupby'][0] in df.columns:
                group_col = structure['groupby'][0]
                value_col = cols[0]
                grouped_data = df.groupby(group_col)[value_col].mean().head(10)
                
                return {
                    'type': 'bar',
                    'x': grouped_data.index.tolist(),
                    'y': grouped_data.values.tolist(),
                    'title': f'Average {value_col} by {group_col}'
                }
            else:
                # Simple bar chart of column means
                means = df[cols].mean()
                return {
                    'type': 'bar',
                    'x': means.index.tolist(),
                    'y': means.values.tolist(),
                    'title': 'Column Averages'
                }
        except Exception:
            return None
    
    def _create_trend_viz_data(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict:
        """Create visualization data for trend analysis"""
        try:
            return {
                'type': 'line',
                'x': df[date_col].dt.strftime('%Y-%m-%d').tolist(),
                'y': df[value_col].tolist(),
                'title': f'{value_col} Trend Over Time'
            }
        except Exception:
            return None
    
    def _create_comparison_viz_data(self, df: pd.DataFrame, group_col: str, value_col: str) -> Dict:
        """Create visualization data for comparison"""
        try:
            grouped = df.groupby(group_col)[value_col].mean().head(10)
            return {
                'type': 'bar',
                'x': grouped.index.tolist(),
                'y': grouped.values.tolist(),
                'title': f'Comparison of {value_col} by {group_col}'
            }
        except Exception:
            return None
    
    def _create_correlation_viz_data(self, corr_matrix: pd.DataFrame) -> Dict:
        """Create visualization data for correlation"""
        try:
            return {
                'type': 'heatmap',
                'z': corr_matrix.values.tolist(),
                'x': corr_matrix.columns.tolist(),
                'y': corr_matrix.index.tolist(),
                'title': 'Correlation Matrix'
            }
        except Exception:
            return None
    
    def _create_ranking_viz_data(self, df: pd.DataFrame, value_col: str) -> Dict:
        """Create visualization data for ranking"""
        try:
            return {
                'type': 'bar',
                'x': list(range(len(df))),
                'y': df[value_col].tolist(),
                'title': f'Ranking by {value_col}'
            }
        except Exception:
            return None
    
    def _create_distribution_viz_data(self, df: pd.DataFrame, value_col: str) -> Dict:
        """Create visualization data for distribution"""
        try:
            return {
                'type': 'histogram',
                'x': df[value_col].tolist(),
                'title': f'Distribution of {value_col}'
            }
        except Exception:
            return None
    
    def _create_general_viz_data(self, df: pd.DataFrame) -> Dict:
        """Create visualization data for general analysis"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                return {
                    'type': 'bar',
                    'x': numeric_cols,
                    'y': df[numeric_cols].mean().tolist(),
                    'title': 'Average Values by Column'
                }
        except Exception:
            return None
    
    def _generate_response(self, query: str, intent: str, analysis_result: Dict, entities: Dict) -> Dict[str, Any]:
        """Generate natural language response with visualization"""
        try:
            result_data = analysis_result.get('result', {})
            
            if 'error' in result_data:
                return {
                    'text': f"I encountered an error while analyzing your data: {result_data['error']}",
                    'visualization': None,
                    'data': result_data
                }
            
            # Generate text response based on intent
            if intent == 'aggregation':
                text = self._generate_aggregation_response(result_data)
            elif intent == 'trend_analysis':
                text = self._generate_trend_response(result_data)
            elif intent == 'comparison':
                text = self._generate_comparison_response(result_data)
            elif intent == 'correlation':
                text = self._generate_correlation_response(result_data)
            elif intent == 'ranking':
                text = self._generate_ranking_response(result_data)
            elif intent == 'distribution':
                text = self._generate_distribution_response(result_data)
            else:
                text = self._generate_general_response(result_data)
            
            # Create visualization
            visualization = self._create_plotly_visualization(analysis_result.get('visualization_data'))
            
            return {
                'text': text,
                'visualization': visualization,
                'data': result_data
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                'text': f"I analyzed your data but encountered an error generating the response: {str(e)}",
                'visualization': None,
                'data': analysis_result.get('result', {})
            }
    
    def _generate_aggregation_response(self, result_data: Dict) -> str:
        """Generate response for aggregation analysis"""
        data = result_data.get('data', {})
        
        if 'summary' in data:
            summary = data['summary']
            response = "Here's a summary of your data:\n\n"
            for col, stats in summary.items():
                response += f"**{col}**: Mean = {stats.get('mean', 0):.2f}, "
                response += f"Total = {stats.get('count', 0)} records\n"
        else:
            response = "I've calculated the requested aggregations for your data. "
            if 'sums' in data:
                response += f"The totals are: {', '.join([f'{k}: {v:.2f}' for k, v in data['sums'].items()])}. "
            if 'averages' in data:
                response += f"The averages are: {', '.join([f'{k}: {v:.2f}' for k, v in data['averages'].items()])}. "
        
        return response
    
    def _generate_trend_response(self, result_data: Dict) -> str:
        """Generate response for trend analysis"""
        data = result_data.get('data', {})
        
        if 'trend_stats' in data:
            stats = data['trend_stats']
            column = data['trend'].get('column', 'the variable')
            
            response = f"Based on the trend analysis of {column}:\n\n"
            response += f"• **Total change**: {stats['total_change']:.2f}\n"
            response += f"• **Percent change**: {stats['percent_change']:.1f}%\n"
            response += f"• **Average value**: {stats['average']:.2f}\n"
            response += f"• **Volatility (std dev)**: {stats['volatility']:.2f}\n"
            
            if stats['percent_change'] > 10:
                response += "\n📈 This shows a strong positive trend!"
            elif stats['percent_change'] < -10:
                response += "\n📉 This shows a declining trend."
            else:
                response += "\n📊 The trend is relatively stable."
        else:
            response = "I've analyzed the trend in your data. Please check the visualization for details."
        
        return response
    
    def _generate_comparison_response(self, result_data: Dict) -> str:
        """Generate response for comparison analysis"""
        data = result_data.get('data', {})
        
        if 'ranking' in data:
            ranking = data['ranking']
            response = f"Here's the comparison analysis:\n\n"
            response += f"**Top performers** (by average):\n"
            for i, (item, value) in enumerate(zip(ranking['by_average'][:3], ranking['values'][:3])):
                response += f"{i+1}. {item}: {value:.2f}\n"
        else:
            response = "I've performed a comparison analysis of your data. Check the visualization for detailed comparisons."
        
        return response
    
    def _generate_correlation_response(self, result_data: Dict) -> str:
        """Generate response for correlation analysis"""
        data = result_data.get('data', {})
        
        if 'strong_correlations' in data:
            correlations = data['strong_correlations']
            response = "Here are the correlation findings:\n\n"
            
            if correlations:
                response += "**Strong correlations found**:\n"
                for corr in correlations[:3]:
                    response += f"• {corr['var1']} and {corr['var2']}: {corr['correlation']:.3f} ({corr['strength']})\n"
            else:
                response += "No strong correlations (>0.5) were found between variables."
        else:
            response = "I've calculated correlations between your variables. Check the correlation matrix in the visualization."
        
        return response
    
    def _generate_ranking_response(self, result_data: Dict) -> str:
        """Generate response for ranking analysis"""
        data = result_data.get('data', {})
        
        if 'ranking' in data:
            ranking = data['ranking']
            column = ranking.get('column', 'values')
            order = ranking.get('order', 'descending')
            
            response = f"Here's the ranking by {column} ({order} order):\n\n"
            for i, (item, value) in enumerate(zip(ranking['items'][:5], ranking['values'][:5])):
                response += f"{i+1}. Item {item}: {value:.2f}\n"
        else:
            response = "I've ranked your data. Please see the visualization for details."
        
        return response
    
    def _generate_distribution_response(self, result_data: Dict) -> str:
        """Generate response for distribution analysis"""
        data = result_data.get('data', {})
        
        if 'statistics' in data:
            stats = data['statistics']
            outliers = data.get('outliers', {})
            
            response = f"Distribution analysis results:\n\n"
            response += f"• **Mean**: {stats['mean']:.2f}\n"
            response += f"• **Median**: {stats['median']:.2f}\n"
            response += f"• **Standard deviation**: {stats['std']:.2f}\n"
            response += f"• **Range**: {stats['min']:.2f} to {stats['max']:.2f}\n"
            
            if outliers.get('count', 0) > 0:
                response += f"\n🔍 Found {outliers['count']} outliers ({outliers['percentage']:.1f}% of data)"
        else:
            response = "I've analyzed the distribution of your data. Check the visualization for details."
        
        return response
    
    def _generate_general_response(self, result_data: Dict) -> str:
        """Generate response for general analysis"""
        data = result_data.get('data', {})
        
        if 'overview' in data:
            overview = data['overview']
            response = f"Here's an overview of your data:\n\n"
            response += f"• **Shape**: {overview['shape'][0]} rows, {overview['shape'][1]} columns\n"
            response += f"• **Columns**: {', '.join(overview['columns'][:5])}"
            if len(overview['columns']) > 5:
                response += f" (and {len(overview['columns']) - 5} more)"
            response += "\n"
            
            missing = sum(overview.get('missing_values', {}).values())
            if missing > 0:
                response += f"• **Missing values**: {missing} total\n"
            else:
                response += "• **Data quality**: No missing values found ✓\n"
        else:
            response = "I've analyzed your data. Please see the results below."
        
        return response
    
    def _create_plotly_visualization(self, viz_data: Optional[Dict]) -> Optional[go.Figure]:
        """Create Plotly visualization from visualization data"""
        if not viz_data:
            return None
        
        try:
            viz_type = viz_data.get('type')
            
            if viz_type == 'bar':
                fig = go.Figure(data=go.Bar(x=viz_data['x'], y=viz_data['y']))
                fig.update_layout(title=viz_data.get('title', 'Bar Chart'))
            
            elif viz_type == 'line':
                fig = go.Figure(data=go.Scatter(x=viz_data['x'], y=viz_data['y'], mode='lines+markers'))
                fig.update_layout(title=viz_data.get('title', 'Line Chart'))
            
            elif viz_type == 'histogram':
                fig = go.Figure(data=go.Histogram(x=viz_data['x']))
                fig.update_layout(title=viz_data.get('title', 'Histogram'))
            
            elif viz_type == 'heatmap':
                fig = go.Figure(data=go.Heatmap(z=viz_data['z'], x=viz_data['x'], y=viz_data['y']))
                fig.update_layout(title=viz_data.get('title', 'Heatmap'))
            
            else:
                return None
            
            # Common layout updates
            fig.update_layout(
                height=500,
                showlegend=True,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return None