# Enhanced Smart Excel Assistant with NLQ and Document Summarization

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import io
from datetime import datetime
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Enhanced imports - handle gracefully if not available
ENHANCED_FEATURES_AVAILABLE = False
try:
    from vector_store import (
        get_simple_vector_store, process_excel_simple, search_simple
    )
    from excel_load_production import get_simple_loader
    from analytics_engine import AdvancedAnalyticsEngine
    from document_summarizer import AdvancedDocumentSummarizer
    from visualization_engine import AdvancedVisualizationEngine
    from nlq_processor import AdvancedNLQProcessor
    from pdf_processor import PDFDocumentProcessor
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    st.error(f"Some enhanced features not available: {e}")
    st.info("Please ensure all required modules are available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title=" RAG AI Analytic Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}
    if 'nlq_history' not in st.session_state:
        st.session_state.nlq_history = []
    if 'document_summaries' not in st.session_state:
        st.session_state.document_summaries = []
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None

def load_css():
    """Load enhanced CSS styling"""
    st.markdown("""
    <style>
    /* Enhanced styling for better UI */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .nlq-input {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .response-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    
    .error-card {
        background: #f8d7da;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    
    .success-card {
        background: #d4edda;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        color: #155724;
    }
    
    .info-card {
        background: #d1ecf1;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
        color: #0c5460;
    }
    
    .metric-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        flex: 1;
    }
    
    .section-header {
        color: #667eea;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        padding: 0.75rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def create_header():
    """Create enhanced application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀  RAG AI Analytic Studio</h1>
        <p><strong>AI-Powered Data Analysis & Document Intelligence Platform</strong></p>
        <p>Advanced Natural Language Queries • Excel/CSV Analysis • PDF Summarization • Interactive Visualizations</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create enhanced sidebar with navigation and status"""
    st.sidebar.title("🎛️ Control Panel")
    
    # Navigation
    st.sidebar.markdown("### 📍 Navigation")
    page = st.sidebar.selectbox(
        "Choose Section",
        ["🏠 Home", "📊 Data Analysis", "🤖 NLQ Assistant", "📄 Document Summarizer", "📈 Advanced Analytics"]
    )
    
    # System Status
    st.sidebar.markdown("### 🔧 System Status")
    
    with st.sidebar.container():
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Data Status
        if st.session_state.processed_data:
            st.success(f"✅ Data Loaded ({len(st.session_state.processed_data)} sheets)")
            for sheet_name, df in st.session_state.processed_data.items():
                st.write(f"📋 **{sheet_name}**: {len(df)} rows, {len(df.columns)} columns")
        else:
            st.warning("⚠️ No data loaded")
        
        # NLQ History
        if st.session_state.nlq_history:
            st.info(f"🗣️ {len(st.session_state.nlq_history)} NLQ queries processed")
        
        # Document Status
        if st.session_state.document_summaries:
            st.info(f"📄 {len(st.session_state.document_summaries)} documents summarized")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    if st.sidebar.button("🗑️ Clear All Data"):
        st.session_state.processed_data = {}
        st.session_state.nlq_history = []
        st.session_state.document_summaries = []
        st.session_state.current_analysis = None
        st.experimental_rerun()
    
    return page

def render_home_page():
    """Render the home page with overview and getting started"""
    st.markdown('<h2 class="section-header">🏠 Welcome to  RAG AI Analytic Studio</h2>', unsafe_allow_html=True)
    
    # Feature overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Excel/CSV Analysis</h3>
            <p>Upload and analyze your spreadsheet data with advanced statistical methods and visualizations.</p>
            <ul>
                <li>Automated data cleaning</li>
                <li>Statistical analysis</li>
                <li>Interactive charts</li>
                <li>Trend detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 Natural Language Queries</h3>
            <p>Ask questions about your data in plain English and get intelligent responses with visualizations.</p>
            <ul>
                <li>Intent recognition</li>
                <li>Automated analysis</li>
                <li>Visual responses</li>
                <li>Context awareness</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📄 Document Summarization</h3>
            <p>Upload PDF documents and get intelligent summaries with key insights extraction.</p>
            <ul>
                <li>PDF text extraction</li>
                <li>Automatic summarization</li>
                <li>Key insights detection</li>
                <li>Multiple summary types</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting Started
    st.markdown('<h3 class="section-header">🚀 Getting Started</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>📋 Step-by-Step Guide:</h4>
        <ol>
            <li><strong>Upload Data</strong>: Go to "Data Analysis" and upload your Excel/CSV file</li>
            <li><strong>Ask Questions</strong>: Use "NLQ Assistant" to ask questions about your data</li>
            <li><strong>Summarize Documents</strong>: Use "Document Summarizer" for PDF analysis</li>
            <li><strong>Advanced Analytics</strong>: Explore detailed statistical analysis</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample queries
    st.markdown('<h4>💡 Sample Questions You Can Ask:</h4>', unsafe_allow_html=True)
    
    sample_queries = [
        "What's the average sales by region?",
        "Show me the trend in revenue over time",
        "Compare performance between Q1 and Q2",
        "Which product has the highest profit margin?",
        "Show me the correlation between price and demand",
        "What are the top 10 customers by sales?"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        st.write(f"{i}. *{query}*")

def render_data_analysis_page():
    """Render the data analysis page with file upload"""
    st.markdown('<h2 class="section-header">📊 Data Analysis</h2>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### 📁 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Choose your Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload Excel (.xlsx, .xls) or CSV files for analysis"
    )
    
    if uploaded_file is not None:
        if st.button("🔄 Process File", type="primary"):
            with st.spinner("Processing your file..."):
                try:
                    if ENHANCED_FEATURES_AVAILABLE:
                        # Use enhanced processing
                        loader = get_simple_loader()
                        result = loader.process_file_simple(uploaded_file)
                        
                        if result.get('success'):
                            st.session_state.processed_data = result.get('data', {})
                            
                            st.markdown("""
                            <div class="success-card">
                                <h4>✅ File Processed Successfully!</h4>
                                <p>Your data has been loaded and is ready for analysis.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show data preview
                            for sheet_name, df in st.session_state.processed_data.items():
                                st.markdown(f"#### 📋 Sheet: {sheet_name}")
                                st.dataframe(df.head(10))
                                st.write(f"**Shape**: {len(df)} rows × {len(df.columns)} columns")
                        else:
                            st.markdown(f"""
                            <div class="error-card">
                                <h4>❌ Processing Failed</h4>
                                <p>{result.get('error_message', 'Unknown error occurred')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Fallback processing
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                            st.session_state.processed_data = {"Sheet1": df}
                        else:
                            data = pd.read_excel(uploaded_file, sheet_name=None)
                            st.session_state.processed_data = data
                        
                        st.success("✅ File processed successfully!")
                        
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-card">
                        <h4>❌ Error Processing File</h4>
                        <p>{str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Data preview section
    if st.session_state.processed_data:
        st.markdown("### 👀 Data Preview")
        
        # Sheet selector
        sheet_names = list(st.session_state.processed_data.keys())
        if len(sheet_names) > 1:
            selected_sheet = st.selectbox("Select Sheet", sheet_names)
        else:
            selected_sheet = sheet_names[0]
        
        df = st.session_state.processed_data[selected_sheet]
        
        # Data metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Rows", len(df))
        with col2:
            st.metric("📋 Columns", len(df.columns))
        with col3:
            st.metric("🔢 Numeric Cols", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("❓ Missing Values", df.isnull().sum().sum())
        
        # Data preview
        st.dataframe(df, use_container_width=True)
        
        # Column information
        with st.expander("📋 Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values,
                'Unique Values': df.nunique().values
            })
            st.dataframe(col_info)

def render_nlq_assistant_page():
    """Render the Natural Language Query assistant page"""
    st.markdown('<h2 class="section-header">🤖 NLQ Assistant</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.markdown("""
        <div class="error-card">
            <h4>⚠️ No Data Available</h4>
            <p>Please upload and process your data first in the "Data Analysis" section.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # NLQ Input
    st.markdown("### 💬 Ask Questions About Your Data")
    
    user_query = st.text_input(
        "Enter your question",
        placeholder="e.g., What are the sales trends by region?",
        help="Ask any question about your data in natural language"
    )
    
    # Sample questions
    with st.expander("💡 Sample Questions"):
        sample_questions = [
            "What's the average revenue by month?",
            "Show me the top 10 customers",
            "Compare sales between regions",
            "What's the correlation between price and quantity?",
            "Show me trends over time",
            "Which products have the highest margin?"
        ]
        
        for question in sample_questions:
            if st.button(f"📝 {question}", key=f"sample_{hash(question)}"):
                user_query = question
                st.experimental_rerun()
    
    # Process query
    if user_query and st.button("🔍 Analyze", type="primary"):
        with st.spinner("Analyzing your question..."):
            try:
                if ENHANCED_FEATURES_AVAILABLE:
                    nlq_processor = AdvancedNLQProcessor()
                    result = nlq_processor.process_query(user_query, st.session_state.processed_data)
                    
                    if result.get('success'):
                        # Add to history
                        st.session_state.nlq_history.append({
                            'query': user_query,
                            'timestamp': datetime.now().isoformat(),
                            'result': result
                        })
                        
                        # Display response
                        response = result.get('response', {})
                        
                        st.markdown(f"""
                        <div class="response-card">
                            <h4>🤖 AI Response</h4>
                            {response.get('text', 'No response generated')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display visualization
                        if response.get('visualization'):
                            st.plotly_chart(response['visualization'], use_container_width=True)
                        
                        # Display raw data if available
                        if response.get('data') and isinstance(response['data'], dict):
                            with st.expander("📊 Raw Data"):
                                st.json(response['data'])
                    
                    else:
                        st.markdown(f"""
                        <div class="error-card">
                            <h4>❌ Analysis Failed</h4>
                            <p>{result.get('error', 'Unknown error occurred')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    st.warning("⚠️ Advanced NLQ processing not available. Basic analysis would be performed here.")
                    
            except Exception as e:
                st.markdown(f"""
                <div class="error-card">
                    <h4>❌ Error Processing Query</h4>
                    <p>{str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Query history
    if st.session_state.nlq_history:
        st.markdown("### 📚 Query History")
        
        for i, entry in enumerate(reversed(st.session_state.nlq_history[-5:]), 1):
            with st.expander(f"Query {len(st.session_state.nlq_history) - i + 1}: {entry['query'][:50]}..."):
                st.write(f"**Query**: {entry['query']}")
                st.write(f"**Time**: {entry['timestamp']}")
                
                if entry['result'].get('success') and entry['result'].get('response'):
                    response = entry['result']['response']
                    st.markdown(response.get('text', 'No response'))
                    
                    if response.get('visualization'):
                        st.plotly_chart(response['visualization'], use_container_width=True)

def render_document_summarizer_page():
    """Render the document summarization page"""
    st.markdown('<h2 class="section-header">📄 Document Summarizer</h2>', unsafe_allow_html=True)
    
    # Document upload section
    st.markdown("### 📁 Upload Document")
    
    uploaded_doc = st.file_uploader(
        "Choose your document",
        type=['pdf', 'txt', 'docx'],
        help="Upload PDF, TXT, or DOCX files for summarization"
    )
    
    if uploaded_doc is not None:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            summary_length = st.selectbox("Summary Length", ["Short", "Medium", "Long"])
            summary_type = st.selectbox("Summary Type", ["Hybrid", "Extractive", "Abstractive"])
        
        if st.button("📝 Generate Summary", type="primary"):
            with st.spinner("Processing document and generating summary..."):
                try:
                    if ENHANCED_FEATURES_AVAILABLE:
                        # Process document
                        pdf_processor = PDFDocumentProcessor()
                        processed_doc = pdf_processor.process_document(uploaded_doc)
                        
                        if processed_doc.get('success'):
                            # Prepare for summarization
                            documents = pdf_processor.prepare_for_summarization(processed_doc['processed_content'])
                            
                            # Generate summary
                            summarizer = AdvancedDocumentSummarizer()
                            
                            # Map summary settings
                            length_map = {"Short": 100, "Medium": 200, "Long": 300}
                            max_length = length_map.get(summary_length, 200)
                            
                            summary_result = summarizer.summarize_documents(
                                documents, 
                                summary_type=summary_type.lower(),
                                max_length=max_length
                            )
                            
                            if not summary_result.get('error'):
                                # Store in session
                                summary_entry = {
                                    'filename': uploaded_doc.name,
                                    'timestamp': datetime.now().isoformat(),
                                    'processed_doc': processed_doc,
                                    'summary_result': summary_result,
                                    'settings': {'length': summary_length, 'type': summary_type}
                                }
                                st.session_state.document_summaries.append(summary_entry)
                                
                                # Display results
                                st.markdown("""
                                <div class="success-card">
                                    <h4>✅ Document Processed Successfully!</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Document metrics
                                metrics = processed_doc.get('metrics', {})
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("📄 Pages", "N/A")
                                with col2:
                                    st.metric("📝 Words", metrics.get('word_count', 0))
                                with col3:
                                    st.metric("⏱️ Read Time", f"{metrics.get('reading_time_minutes', 0)} min")
                                with col4:
                                    st.metric("🧠 Complexity", metrics.get('complexity_score', 0))
                                
                                # Summary display
                                summaries = summary_result.get('summaries', {})
                                
                                if summary_type.lower() in summaries:
                                    summary_text = summaries[summary_type.lower()].get('summary', '')
                                    
                                    st.markdown(f"""
                                    <div class="response-card">
                                        <h4>📋 {summary_type} Summary</h4>
                                        <p>{summary_text}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Key insights
                                key_insights = summary_result.get('key_insights', [])
                                if key_insights:
                                    st.markdown("### 🔍 Key Insights")
                                    for insight in key_insights:
                                        st.write(f"**{insight.get('type', '').title()}**: {insight.get('description', '')}")
                                        if insight.get('examples'):
                                            st.write(f"Examples: {', '.join(str(ex) for ex in insight['examples'][:3])}")
                                
                                # Full text preview
                                with st.expander("📖 Full Text Preview"):
                                    st.text_area(
                                        "Extracted Text", 
                                        processed_doc.get('raw_text', '')[:2000] + "...", 
                                        height=300
                                    )
                            
                            else:
                                st.markdown(f"""
                                <div class="error-card">
                                    <h4>❌ Summarization Failed</h4>
                                    <p>{summary_result.get('error', 'Unknown error')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        else:
                            st.markdown(f"""
                            <div class="error-card">
                                <h4>❌ Document Processing Failed</h4>
                                <p>{processed_doc.get('error', 'Unknown error')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    else:
                        st.warning("⚠️ Advanced document processing not available.")
                        
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-card">
                        <h4>❌ Error Processing Document</h4>
                        <p>{str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Document history
    if st.session_state.document_summaries:
        st.markdown("### 📚 Document History")
        
        for i, entry in enumerate(reversed(st.session_state.document_summaries[-3:]), 1):
            with st.expander(f"📄 {entry['filename']} - {entry['settings']['type']} Summary"):
                st.write(f"**File**: {entry['filename']}")
                st.write(f"**Processed**: {entry['timestamp']}")
                st.write(f"**Settings**: {entry['settings']['length']} length, {entry['settings']['type']} type")
                
                # Display summary
                summary_result = entry.get('summary_result', {})
                summaries = summary_result.get('summaries', {})
                summary_type = entry['settings']['type'].lower()
                
                if summary_type in summaries:
                    summary_text = summaries[summary_type].get('summary', 'No summary available')
                    st.markdown(f"**Summary**: {summary_text}")

def render_advanced_analytics_page():
    """Render advanced analytics page with detailed analysis"""
    st.markdown('<h2 class="section-header">📈 Advanced Analytics</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.markdown("""
        <div class="error-card">
            <h4>⚠️ No Data Available</h4>
            <p>Please upload and process your data first in the "Data Analysis" section.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Sheet selector
    sheet_names = list(st.session_state.processed_data.keys())
    if len(sheet_names) > 1:
        selected_sheet = st.selectbox("Select Sheet for Analysis", sheet_names)
    else:
        selected_sheet = sheet_names[0]
    
    df = st.session_state.processed_data[selected_sheet]
    
    # Analysis type selector
    analysis_type = st.selectbox(
        "Choose Analysis Type",
        ["Statistical Summary", "Correlation Analysis", "Distribution Analysis", "Trend Analysis"]
    )
    
    if st.button("🔬 Run Analysis", type="primary"):
        with st.spinner(f"Running {analysis_type}..."):
            try:
                if analysis_type == "Statistical Summary":
                    # Statistical summary
                    numeric_df = df.select_dtypes(include=[np.number])
                    
                    if not numeric_df.empty:
                        st.markdown("### 📊 Statistical Summary")
                        st.dataframe(numeric_df.describe())
                        
                        # Visualizations for each numeric column
                        for col in numeric_df.columns:
                            st.markdown(f"#### 📈 Distribution: {col}")
                            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No numeric columns found for statistical analysis.")
                
                elif analysis_type == "Correlation Analysis":
                    numeric_df = df.select_dtypes(include=[np.number])
                    
                    if len(numeric_df.columns) >= 2:
                        st.markdown("### 🔗 Correlation Analysis")
                        
                        # Correlation matrix
                        corr_matrix = numeric_df.corr()
                        
                        # Heatmap
                        fig = px.imshow(corr_matrix, title="Correlation Heatmap", aspect="auto")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Correlation table
                        st.dataframe(corr_matrix)
                    else:
                        st.warning("Need at least 2 numeric columns for correlation analysis.")
                
                elif analysis_type == "Distribution Analysis":
                    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                    
                    if numeric_columns:
                        st.markdown("### 📊 Numeric Distributions")
                        for col in numeric_columns[:4]:  # Limit to first 4 columns
                            fig = px.box(df, y=col, title=f"Box Plot: {col}")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if categorical_columns:
                        st.markdown("### 📋 Categorical Distributions")
                        for col in categorical_columns[:3]:  # Limit to first 3 columns
                            value_counts = df[col].value_counts().head(10)
                            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                       title=f"Value Counts: {col}")
                            st.plotly_chart(fig, use_container_width=True)
                
                elif analysis_type == "Trend Analysis":
                    # Find date columns
                    date_columns = []
                    for col in df.columns:
                        if 'date' in col.lower() or 'time' in col.lower():
                            date_columns.append(col)
                    
                    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if date_columns and numeric_columns:
                        st.markdown("### 📈 Trend Analysis")
                        
                        date_col = st.selectbox("Select Date Column", date_columns)
                        value_col = st.selectbox("Select Value Column", numeric_columns)
                        
                        if st.button("Generate Trend Chart"):
                            fig = px.line(df, x=date_col, y=value_col, 
                                        title=f"Trend: {value_col} over {date_col}")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Trend analysis requires date and numeric columns.")
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-card">
                    <h4>❌ Analysis Failed</h4>
                    <p>{str(e)}</p>
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Initialize
    initialize_session_state()
    load_css()
    create_header()
    
    # Sidebar navigation
    current_page = create_sidebar()
    
    # Render appropriate page
    if current_page == "🏠 Home":
        render_home_page()
    elif current_page == "📊 Data Analysis":
        render_data_analysis_page()
    elif current_page == "🤖 NLQ Assistant":
        render_nlq_assistant_page()
    elif current_page == "📄 Document Summarizer":
        render_document_summarizer_page()
    elif current_page == "📈 Advanced Analytics":
        render_advanced_analytics_page()
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit •  RAG AI Analytic Studio v2.0")

if __name__ == "__main__":
    main()