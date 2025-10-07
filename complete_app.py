# Complete Enhanced  RAG AI Analytic Studio

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

# Enhanced imports with fallback handling
ENHANCED_FEATURES_AVAILABLE = False
try:
    from vector_store import (
        get_simple_vector_store, process_excel_simple, search_simple
    )
    from excel_load_production import get_simple_loader
    from ollama_integration import OllamaRAGSystem  # NEW
    from analytics_engine import AdvancedAnalyticsEngine
    from document_summarizer import AdvancedDocumentSummarizer
    from visualization_engine import AdvancedVisualizationEngine
    from nlq_processor import AdvancedNLQProcessor
    from pdf_processor import PDFDocumentProcessor
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    st.warning(f"Some enhanced features not available: {e}")
    st.info("Please ensure all required modules are installed")
import streamlit as st
import os

UPLOAD_DIR ="C:\\Users\\laksh\\OneDrive\Desktop\\smart excel database"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

def load_css():
    """Load enhanced CSS styling"""
    st.markdown("""
    <style>
    /* Enhanced styling for better UI */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .nlq-container {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .response-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .error-card {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #dc3545;
        color: #721c24;
    }
    
    .success-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
        color: #155724;
    }
    
    .info-card {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #17a2b8;
        color: #0c5460;
    }
    
    .chat-message {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        margin-right: 2rem;
    }
    
    .section-header {
        color: #667eea;
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def create_header():
    """Create enhanced application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 RAG AI Analytic Studio </h1>
        <p><strong>Advanced AI-Powered Data Analysis & Document Intelligence Platform</strong></p>
        <p>Natural Language Queries • Excel/CSV Analysis • PDF/Word Summarization • Interactive Visualizations</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create enhanced sidebar with navigation and status"""
    st.sidebar.title("🎛️ Control Panel")
    
    # Navigation
    st.sidebar.markdown("### 📍 Navigation")
    page = st.sidebar.selectbox(
        "Choose Section",
        ["🏠 Home", "📊 Data Analysis", "💬 AI Assistant", "📄 Document Summarizer", "📈 Advanced Analytics"]
    )
    
    # System Status
    st.sidebar.markdown("### 🔧 System Status")
    
    with st.sidebar.container():
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Enhanced Features Status
        if ENHANCED_FEATURES_AVAILABLE:
            st.success("✅ All features active")
        else:
            st.warning("⚠️ Limited functionality")
        
        # Data Status
        if st.session_state.processed_data:
            st.success(f"📊 Data loaded ({len(st.session_state.processed_data)} sheets)")
            for sheet_name, df in st.session_state.processed_data.items():
                st.write(f"📋 **{sheet_name}**: {len(df)} rows, {len(df.columns)} columns")
        else:
            st.info("📊 No data loaded")
        
        # Chat History
        if st.session_state.chat_messages:
            st.info(f"💬 {len(st.session_state.chat_messages)} chat messages")
        
        # Document Status
        if st.session_state.document_summaries:
            st.info(f"📄 {len(st.session_state.document_summaries)} documents processed")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    if st.sidebar.button("🗑️ Clear All Data"):
        st.session_state.processed_data = {}
        st.session_state.chat_messages = []
        st.session_state.document_summaries = []
        st.session_state.current_analysis = None
        st.experimental_rerun()
    
    return page

def render_home_page():
    """Render the enhanced home page"""
    st.markdown('<h2 class="section-header">🏠 Welcome to  RAG AI Analytic Studio</h2>', unsafe_allow_html=True)
    
    # Feature overview with enhanced cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 AI-Powered Analysis</h3>
            <p>Ask questions about your data in natural language and get intelligent responses with visualizations.</p>
            <ul>
                <li>Intent recognition & NLQ processing</li>
                <li>Automated statistical analysis</li>
                <li>Smart visualizations</li>
                <li>Contextual insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Excel/CSV Analysis</h3>
            <p>Upload and analyze spreadsheet data with advanced statistical methods and interactive charts.</p>
            <ul>
                <li>Automated data cleaning</li>
                <li>Advanced analytics engine</li>
                <li>Trend & correlation analysis</li>
                <li>Predictive modeling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📄 Document Intelligence</h3>
            <p>Upload PDF and Word documents for intelligent summarization and key insights extraction.</p>
            <ul>
                <li>PDF & DOCX text extraction</li>
                <li>Multi-approach summarization</li>
                <li>Key insights detection</li>
                <li>Entity extraction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting Started Guide
    st.markdown('<h3 class="section-header">🚀 Getting Started</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>📋 Quick Start Guide:</h4>
        <ol>
            <li><strong>📊 Data Analysis</strong>: Upload Excel/CSV files for comprehensive data analysis</li>
            <li><strong>💬 AI Assistant</strong>: Ask questions about your data in plain English</li>
            <li><strong>📄 Document Summarizer</strong>: Upload PDF/Word documents for intelligent summarization</li>
            <li><strong>📈 Advanced Analytics</strong>: Explore detailed statistical analysis and predictions</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample queries showcase
    st.markdown('<h4>💡 What You Can Ask:</h4>', unsafe_allow_html=True)
    
    sample_queries = [
        "📈 What's the sales trend over the last 12 months?",
        "📊 Show me the top 10 customers by revenue",
        "🔍 Compare performance between different regions",
        "📉 Which products have declining sales?",
        "🔗 What's the correlation between marketing spend and revenue?",
        "🎯 Predict next quarter's sales based on current trends"
    ]
    
    for query in sample_queries:
        st.write(f"• *{query}*")

def render_data_analysis_page():
    """Render the enhanced data analysis page"""
    st.markdown('<h2 class="section-header">📊 Data Analysis</h2>', unsafe_allow_html=True)
    
    # File upload section with enhanced UI
    st.markdown("### 📁 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Choose your Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload Excel (.xlsx, .xls) or CSV files for comprehensive analysis"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📁 **{uploaded_file.name}** ({file_size:.2f} MB)")
        
        with col2:
            if st.button("🔄 Process File", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing your file..."):
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
                                    <p>Your data has been loaded and is ready for AI-powered analysis.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Enhanced data preview
                                for sheet_name, df in st.session_state.processed_data.items():
                                    with st.expander(f"📋 **{sheet_name}** - {len(df)} rows × {len(df.columns)} columns"):
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Rows", len(df))
                                        with col2:
                                            st.metric("Columns", len(df.columns))
                                        with col3:
                                            st.metric("Numeric Cols", len(df.select_dtypes(include=[np.number]).columns))
                                        with col4:
                                            st.metric("Missing Values", df.isnull().sum().sum())
                                        
                                        st.dataframe(df.head(10), use_container_width=True)
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
    
    # Enhanced data overview
    if st.session_state.processed_data:
        st.markdown("### 👀 Data Overview & Quick Stats")
        
        # Overall statistics
        total_rows = sum(len(df) for df in st.session_state.processed_data.values())
        total_cols = sum(len(df.columns) for df in st.session_state.processed_data.values())
        total_missing = sum(df.isnull().sum().sum() for df in st.session_state.processed_data.values())
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-box"><h3>📊</h3><h2>{}</h2><p>Total Rows</p></div>'.format(total_rows), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-box"><h3>📋</h3><h2>{}</h2><p>Total Columns</p></div>'.format(total_cols), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-box"><h3>📈</h3><h2>{}</h2><p>Sheets</p></div>'.format(len(st.session_state.processed_data)), unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-box"><h3>❓</h3><h2>{}</h2><p>Missing Values</p></div>'.format(total_missing), unsafe_allow_html=True)

def render_ai_assistant_page():
    """Render the enhanced AI assistant page with chat interface"""
    st.markdown('<h2 class="section-header">💬 AI Assistant</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.markdown("""
        <div class="error-card">
            <h4>⚠️ No Data Available</h4>
            <p>Please upload and process your data first in the "Data Analysis" section.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Enhanced chat interface
    st.markdown("### 💬 Chat with Your Data")
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message ai-message"><strong>AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                
                # Display visualization if available
                if message.get("visualization"):
                    st.plotly_chart(message["visualization"], use_container_width=True)
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input(
            "Ask me anything about your data...",
            placeholder="e.g., What are the sales trends by region?",
            key="chat_input"
        )
    
    with col2:
        send_button = st.button("🚀 Send", type="primary", use_container_width=True)
    
    # Sample questions for quick access
    st.markdown("### 💡 Quick Questions")
    sample_questions = [
        "📈 Show me trends over time",
        "📊 What are the summary statistics?",
        "🔍 Find correlations in the data",
        "🎯 Which are the top performers?",
        "📉 Identify any anomalies",
        "🔮 Can you make predictions?"
    ]
    
    question_cols = st.columns(3)
    for i, question in enumerate(sample_questions):
        with question_cols[i % 3]:
            if st.button(question, key=f"sample_q_{i}", use_container_width=True):
                user_query = question
                send_button = True
    
    # Process user query
    if (user_query and send_button) or (user_query and st.session_state.get("auto_send")):
        # Add user message to chat
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now().isoformat()
        })
        
        with st.spinner("🤖 AI is analyzing your data..."):
            try:
                if ENHANCED_FEATURES_AVAILABLE:
                    nlq_processor = AdvancedNLQProcessor()
                    result = nlq_processor.process_query(user_query, st.session_state.processed_data)
                    
                    if result.get('success'):
                        response = result.get('response', {})
                        
                        # Add AI response to chat
                        ai_message = {
                            "role": "assistant",
                            "content": response.get('text', 'I processed your query but have no response to show.'),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Add visualization if available
                        if response.get('visualization'):
                            ai_message["visualization"] = response['visualization']
                        
                        st.session_state.chat_messages.append(ai_message)
                        
                        # Store in NLQ history
                        st.session_state.nlq_history.append({
                            'query': user_query,
                            'timestamp': datetime.now().isoformat(),
                            'result': result
                        })
                    
                    else:
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": f"I'm sorry, I encountered an error: {result.get('error', 'Unknown error occurred')}",
                            "timestamp": datetime.now().isoformat()
                        })
                
                else:
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": "I'm running in basic mode. Advanced NLQ processing is not available. Please ensure all required modules are installed.",
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except Exception as e:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": f"I encountered an error processing your query: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Clear input and refresh
        st.experimental_rerun()

def render_document_summarizer_page():
    """Render the enhanced document summarization page"""
    st.markdown('<h2 class="section-header">📄 Document Summarizer</h2>', unsafe_allow_html=True)
    
    # Enhanced document upload section
    st.markdown("### 📁 Upload Document for AI Summarization")
    
    uploaded_doc = st.file_uploader(
        "Choose your document",
        type=['pdf', 'txt', 'docx'],
        help="Upload PDF, TXT, or DOCX files for intelligent summarization and analysis"
    )
    
    if uploaded_doc is not None:
        # Enhanced settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            summary_length = st.selectbox("📏 Summary Length", ["Short (100 words)", "Medium (200 words)", "Long (300 words)"])
        with col2:
            summary_type = st.selectbox("🎯 Summary Type", ["Hybrid (Best)", "Extractive", "Abstractive"])
        with col3:
            extract_insights = st.checkbox("🔍 Extract Key Insights", value=True)
        
        if st.button("📝 Analyze & Summarize", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is processing your document..."):
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
                            length_map = {"Short (100 words)": 100, "Medium (200 words)": 200, "Long (300 words)": 300}
                            max_length = length_map.get(summary_length, 200)
                            clean_type = summary_type.split(" ")[0].lower()
                            
                            summary_result = summarizer.summarize_documents(
                                documents, 
                                summary_type=clean_type,
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
                                
                                # Display results with enhanced UI
                                st.markdown("""
                                <div class="success-card">
                                    <h4>✅ Document Processed Successfully!</h4>
                                    <p>Your document has been analyzed and summarized by our AI.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Enhanced document metrics
                                metrics = processed_doc.get('metrics', {})
                                col1, col2, col3, col4, col5 = st.columns(5)
                                
                                with col1:
                                    st.markdown('<div class="metric-box"><h3>📄</h3><h2>{}</h2><p>File Type</p></div>'.format(processed_doc.get('file_type', 'Unknown')), unsafe_allow_html=True)
                                with col2:
                                    st.markdown('<div class="metric-box"><h3>📝</h3><h2>{:,}</h2><p>Words</p></div>'.format(metrics.get('word_count', 0)), unsafe_allow_html=True)
                                with col3:
                                    st.markdown('<div class="metric-box"><h3>📖</h3><h2>{:,}</h2><p>Characters</p></div>'.format(metrics.get('character_count', 0)), unsafe_allow_html=True)
                                with col4:
                                    st.markdown('<div class="metric-box"><h3>⏱️</h3><h2>{}</h2><p>Read Time (min)</p></div>'.format(metrics.get('reading_time_minutes', 0)), unsafe_allow_html=True)
                                with col5:
                                    complexity_labels = {1: "Low", 2: "Medium", 3: "High"}
                                    complexity = complexity_labels.get(metrics.get('complexity_score', 1), "Unknown")
                                    st.markdown('<div class="metric-box"><h3>🧠</h3><h2>{}</h2><p>Complexity</p></div>'.format(complexity), unsafe_allow_html=True)
                                
                                # Enhanced summary display
                                summaries = summary_result.get('summaries', {})
                                
                                if clean_type in summaries:
                                    summary_text = summaries[clean_type].get('summary', '')
                                    
                                    st.markdown(f"""
                                    <div class="response-card">
                                        <h4>📋 {summary_type} Summary</h4>
                                        <p style="font-size: 1.1em; line-height: 1.6;">{summary_text}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Enhanced key insights
                                if extract_insights and summary_result.get('key_insights'):
                                    st.markdown("### 🔍 Key Insights & Analysis")
                                    
                                    insights_by_type = {}
                                    for insight in summary_result['key_insights']:
                                        insight_type = insight.get('type', 'general')
                                        if insight_type not in insights_by_type:
                                            insights_by_type[insight_type] = []
                                        insights_by_type[insight_type].append(insight)
                                    
                                    for insight_type, insights in insights_by_type.items():
                                        with st.expander(f"📊 {insight_type.title()} Insights"):
                                            for insight in insights:
                                                st.write(f"**{insight.get('description', '')}**")
                                                if insight.get('examples'):
                                                    st.write("Examples:", ", ".join(str(ex) for ex in insight['examples'][:5]))
                                                st.write("---")
                                
                                # Enhanced text preview
                                with st.expander("📖 Full Text Preview"):
                                    preview_text = processed_doc.get('raw_text', '')[:5000]
                                    st.text_area(
                                        "Document Content", 
                                        preview_text + ("..." if len(processed_doc.get('raw_text', '')) > 5000 else ""),
                                        height=400
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
                        st.markdown("""
                        <div class="error-card">
                            <h4>⚠️ Advanced Features Not Available</h4>
                            <p>Document processing requires additional dependencies. Please ensure all modules are installed.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-card">
                        <h4>❌ Error Processing Document</h4>
                        <p>{str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Enhanced document history
    if st.session_state.document_summaries:
        st.markdown("### 📚 Recent Document Summaries")
        
        for i, entry in enumerate(reversed(st.session_state.document_summaries[-5:]), 1):
            with st.expander(f"📄 {entry['filename']} - {entry['settings']['type']} ({entry['settings']['length']})"):
                st.write(f"**Processed**: {entry['timestamp'][:19].replace('T', ' ')}")
                
                # Display summary
                summary_result = entry.get('summary_result', {})
                summaries = summary_result.get('summaries', {})
                summary_type = entry['settings']['type'].split(" ")[0].lower()
                
                if summary_type in summaries:
                    summary_text = summaries[summary_type].get('summary', 'No summary available')
                    st.markdown(f"**Summary**: {summary_text}")
                
                # Show metrics if available
                if entry.get('processed_doc', {}).get('metrics'):
                    metrics = entry['processed_doc']['metrics']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Words", f"{metrics.get('word_count', 0):,}")
                    with col2:
                        st.metric("Reading Time", f"{metrics.get('reading_time_minutes', 0)} min")
                    with col3:
                        complexity = {1: "Low", 2: "Medium", 3: "High"}.get(metrics.get('complexity_score', 1), "Unknown")
                        st.metric("Complexity", complexity)

def render_advanced_analytics_page():
    """Render enhanced advanced analytics page"""
    st.markdown('<h2 class="section-header">📈 Advanced Analytics</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_data:
        st.markdown("""
        <div class="error-card">
            <h4>⚠️ No Data Available</h4>
            <p>Please upload and process your data first in the "Data Analysis" section.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Enhanced analytics interface
    st.markdown("### 🔬 Choose Your Analysis")
    
    # Sheet selector
    sheet_names = list(st.session_state.processed_data.keys())
    if len(sheet_names) > 1:
        selected_sheet = st.selectbox("📊 Select Dataset", sheet_names)
    else:
        selected_sheet = sheet_names[0]
        st.info(f"📊 Analyzing dataset: **{selected_sheet}**")
    
    df = st.session_state.processed_data[selected_sheet]
    
    # Analysis type selector with descriptions
    analysis_options = {
        "📊 Statistical Summary": "Comprehensive descriptive statistics and data distribution analysis",
        "🔗 Correlation Analysis": "Discover relationships between variables with correlation matrices",
        "📈 Trend Analysis": "Time-series analysis with trend detection and forecasting",
        "🔍 Outlier Detection": "Identify anomalies and outliers in your data",
        "📉 Distribution Analysis": "Analyze data distributions and patterns"
    }
    
    selected_analysis = st.selectbox("🎯 Analysis Type", list(analysis_options.keys()))
    st.info(f"ℹ️ {analysis_options[selected_analysis]}")
    
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        with st.spinner(f"🔬 Running {selected_analysis}..."):
            try:
                analysis_type = selected_analysis.split(" ", 1)[1]  # Remove emoji
                
                if analysis_type == "Statistical Summary":
                    # Enhanced statistical summary
                    numeric_df = df.select_dtypes(include=[np.number])
                    
                    if not numeric_df.empty:
                        st.markdown("### 📊 Statistical Summary")
                        
                        # Enhanced summary statistics
                        summary_stats = numeric_df.describe()
                        st.dataframe(summary_stats.round(3), use_container_width=True)
                        
                        # Visualizations for each numeric column
                        st.markdown("### 📈 Data Distributions")
                        
                        cols = st.columns(min(2, len(numeric_df.columns)))
                        for idx, col in enumerate(numeric_df.columns[:4]):  # Limit to 4 charts
                            with cols[idx % 2]:
                                fig = px.histogram(df, x=col, title=f"Distribution of {col}",
                                                 marginal="box", hover_data=df.columns)
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ No numeric columns found for statistical analysis.")
                
                elif analysis_type == "Correlation Analysis":
                    numeric_df = df.select_dtypes(include=[np.number])
                    
                    if len(numeric_df.columns) >= 2:
                        st.markdown("### 🔗 Correlation Analysis")
                        
                        # Enhanced correlation matrix
                        corr_matrix = numeric_df.corr()
                        
                        # Interactive correlation heatmap
                        fig = px.imshow(corr_matrix, 
                                      title="Interactive Correlation Matrix",
                                      aspect="auto",
                                      color_continuous_scale="RdBu_r",
                                      zmin=-1, zmax=1)
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Strong correlations summary
                        strong_correlations = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr_val = corr_matrix.iloc[i, j]
                                if abs(corr_val) > 0.5:  # Strong correlation threshold
                                    strong_correlations.append({
                                        'Variable 1': corr_matrix.columns[i],
                                        'Variable 2': corr_matrix.columns[j],
                                        'Correlation': round(corr_val, 3),
                                        'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                                    })
                        
                        if strong_correlations:
                            st.markdown("### 🎯 Strong Correlations Found")
                            st.dataframe(pd.DataFrame(strong_correlations), use_container_width=True)
                        else:
                            st.info("ℹ️ No strong correlations (>0.5) found between variables.")
                    
                    else:
                        st.warning("⚠️ Need at least 2 numeric columns for correlation analysis.")
                
                elif analysis_type == "Distribution Analysis":
                    st.markdown("### 📊 Data Distribution Analysis")
                    
                    # Numeric distributions
                    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                    
                    if numeric_columns:
                        st.markdown("#### 📈 Numeric Variable Distributions")
                        selected_numeric = st.multiselect("Select numeric columns to analyze:", numeric_columns, default=numeric_columns[:3])
                        
                        for col in selected_numeric:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_hist = px.histogram(df, x=col, title=f"Histogram: {col}",
                                                      marginal="rug", hover_data=df.columns)
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            with col2:
                                fig_box = px.box(df, y=col, title=f"Box Plot: {col}")
                                st.plotly_chart(fig_box, use_container_width=True)
                    
                    if categorical_columns:
                        st.markdown("#### 📋 Categorical Variable Distributions")
                        selected_categorical = st.selectbox("Select categorical column:", categorical_columns)
                        
                        if selected_categorical:
                            value_counts = df[selected_categorical].value_counts().head(20)
                            fig = px.bar(x=value_counts.index, y=value_counts.values,
                                       title=f"Value Counts: {selected_categorical}")
                            st.plotly_chart(fig, use_container_width=True)
                
                elif analysis_type == "Outlier Detection":
                    st.markdown("### 🔍 Outlier Detection Analysis")
                    
                    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_columns:
                        selected_col = st.selectbox("Select column for outlier detection:", numeric_columns)
                        
                        if selected_col:
                            # IQR method for outlier detection
                            Q1 = df[selected_col].quantile(0.25)
                            Q3 = df[selected_col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Outliers", len(outliers))
                            with col2:
                                st.metric("Outlier Percentage", f"{len(outliers)/len(df)*100:.2f}%")
                            
                            # Visualization
                            fig = px.box(df, y=selected_col, title=f"Outlier Detection: {selected_col}")
                            fig.add_hline(y=lower_bound, line_dash="dash", line_color="red", 
                                        annotation_text="Lower Threshold")
                            fig.add_hline(y=upper_bound, line_dash="dash", line_color="red",
                                        annotation_text="Upper Threshold")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            if len(outliers) > 0:
                                st.markdown("#### 🎯 Detected Outliers")
                                st.dataframe(outliers, use_container_width=True)
                    else:
                        st.warning("⚠️ No numeric columns available for outlier detection.")
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-card">
                    <h4>❌ Analysis Failed</h4>
                    <p>{str(e)}</p>
                </div>
                """, unsafe_allow_html=True)

def main():
    """Enhanced main application function"""
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
    elif current_page == "💬 AI Assistant":
        render_ai_assistant_page()
    elif current_page == "📄 Document Summarizer":
        render_document_summarizer_page()
    elif current_page == "📈 Advanced Analytics":
        render_advanced_analytics_page()
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 12px; margin-top: 2rem;">
        <h4>🚀  RAG AI Analytic Studio v2.0</h4>
        <p>Built with ❤️ using Streamlit • Advanced AI-Powered Data Analysis & Document Intelligence</p>
        <p><em>Upload your data, ask questions, get intelligent insights!</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {str(e)}")
        st.info("Please check your setup and try refreshing the page.")
        logger.error(f"Application error: {e}")