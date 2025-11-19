
import os
import sys

# Ensure project root is on sys.path so local modules are importable
_PROJECT_ROOT = os.path.dirname(__file__)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import pandas as pd
import json
import tempfile
import logging
from pathlib import Path
from io import StringIO, BytesIO
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

from config import UPLOAD_DIR
from utils.storage_manager import StorageManager
from utils.excel_handler import ExcelHandler
from utils.document_handler import DocumentHandler
from utils.llm_handler import LLMHandler
from utils.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Suppress ChromaDB warnings
logging.getLogger("chromadb").setLevel(logging.WARNING)

# ============= PAGE CONFIG & UI HELPERS =============

st.set_page_config(
    page_title="🚀 RAG AI Analytic Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css():
    """Load enhanced CSS styling"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }

    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }

    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }

    .chat-message {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }

    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
    }

    .upload-area {
        border: 2px dashed #a78bfa;
        padding: 1.5rem;
        border-radius: 12px;
        background: rgba(103, 58, 183, 0.05);
        text-align: center;
        transition: all 0.2s ease;
        margin-bottom: 1rem;
    }

    .upload-area:hover {
        border-color: #7c3aed;
        background: rgba(103, 58, 183, 0.1);
    }

    .upload-area span {
        display: block;
        color: #6b46c1;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 1.15rem;
        font-weight: 600;
        padding: 0.85rem 1.75rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(102, 126, 234, 0.15);
        color: #4c1d95;
        border-radius: 30px;
    }
    </style>
    """, unsafe_allow_html=True)


def create_header():
    """Create enhanced application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 RAG AI Analytic Studio</h1>
        <p><strong>Advanced AI-Powered Data Analysis & Document Intelligence Platform</strong></p>
        <p>Natural Language Queries • Excel Analysis • Document Summarization</p>
    </div>
    """, unsafe_allow_html=True)


def create_feature_cards():
    """Create feature description cards"""
    col1, col2, col3,col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 Natural Language Queries</h4>
            <p>Ask questions about your Excel data and get AI-powered insights.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Excel Intelligence</h4>
            <p>Upload and analyze Excel files with automatic data processing.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>📄 Document Summarizer</h4>
            <p>Upload PDFs and generate AI-powered summaries instantly.</p>
        </div>
        """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="feature-card">
                <h4>📑 Report Generator</h4>
                <p>Generate AI-powered reports from your data and documents</p>
            </div>
            """, unsafe_allow_html=True)

def display_data_info(df):
    """Display data information in metrics"""
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>{}</h3>
                <p>Total Rows</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>{}</h3>
                <p>Total Columns</p>
            </div>
            """.format(len(df.columns)), unsafe_allow_html=True)

        with col3:
            missing_values = df.isnull().sum().sum()
            st.markdown("""
            <div class="metric-card">
                <h3>{}</h3>
                <p>Missing Values</p>
            </div>
            """.format(missing_values), unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>{}</h3>
                <p>Data Type Issues</p>
            </div>
            """.format(0), unsafe_allow_html=True)


# ============= INITIALIZE SESSION STATE =============

if "llm_handler" not in st.session_state:
    try:
        st.session_state.llm_handler = LLMHandler()
    except ImportError as e:
        st.error(f"❌ Missing Package: {str(e)}")
        st.code("pip install langchain-ollama", language="bash")
        st.stop()
    except ConnectionError as e:
        st.error(f"❌ Connection Error: {str(e)}")
        st.info("💡 **Solution:** Start Ollama server with: `ollama serve`")
        st.stop()
    except ValueError as e:
        st.error(f"❌ Model Error: {str(e)}")
        st.info("💡 **Solution:** Download the required models")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error initializing LLM: {str(e)}")
        st.stop()

if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "loaded_sheet" not in st.session_state:
    st.session_state.loaded_sheet = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = {}  # Store document summaries: {filename: summary}

# ============= LOAD UI & CREATE HEADER =============

load_css()
create_header()
create_feature_cards()

# ============= MAIN NAVIGATION TABS =============

# Custom CSS to increase tab text size
st.markdown("""
    <style>
    /* Increase tab text size */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 20px !important;
        font-weight: 600 !important;
    }
    
    /* Optional: Increase tab height for better spacing */
    .stTabs [data-baseweb="tab-list"] button {
        height: 60px !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    
    /* Optional: Make active tab more prominent */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
        font-size: 22px !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ============= MAIN NAVIGATION TABS =============
tab_excel, tab_doc, tab_reports = st.tabs([
    "📊 Excel Analysis", 
    "📄 Document Summarizer",
    "📑 Generate Reports"
])

# ============= SIDEBAR CONTENT =============

with st.sidebar:
    st.header("ℹ️ Platform Info")
    st.markdown("""
    - 🤖 Powered by local **Ollama** (Llama 3.2)
    - 📊 Vector store: **ChromaDB**
    - ⚙️ Built on **Streamlit**
    - 💾 Secure local storage
    """)

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Use the **Excel Analysis** tab for spreadsheet insights  
    - Use the **Document Summarizer** tab for PDF intelligence  
    - Generated console output now appears directly in the UI  
    """)


# ============= EXCEL ANALYSIS MODE =============

with tab_excel:
    
    st.markdown("---")
    st.subheader("📊 Excel Data Analysis with Natural Language Queries")

    recent_files = StorageManager.get_uploaded_files("excel")
    uploaded_file = None
    selected_recent = "-- Select --"

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload Excel File")
        st.markdown("""
        <div class="upload-area">
            <span>Drag & drop Excel file here</span>
            or click to browse
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag & drop Excel file or click to browse",
            type=["xlsx", "xls"],
            key="excel_uploader",
            label_visibility="collapsed"
        )
        st.caption("Supports .xlsx & .xls (up to 50 MB). Data stays on your device.")
    
    with col2:
        st.subheader("📂 Recent Files")
        if recent_files:
            selected_recent = st.selectbox(
                "Quick select:",
                ["-- Select --"] + [f["name"] for f in recent_files],
                key="file_selector"
            )
        else:
            st.info("No recent files yet. Upload an Excel file to see it here.")

    # Handle file upload
    file_path = None
    
    if uploaded_file:
        file_path = StorageManager.save_uploaded_file(uploaded_file, "excel")
        st.session_state.current_file = file_path
        st.session_state.current_df = None
        st.session_state.loaded_sheet = None
    elif st.session_state.current_file and os.path.exists(st.session_state.current_file):
        file_path = st.session_state.current_file

    if recent_files and selected_recent != "-- Select --":
        for f in recent_files:
            if f["name"] == selected_recent:
                st.session_state.current_file = f["path"]
                file_path = f["path"]
                st.session_state.current_df = None
                st.session_state.loaded_sheet = None
                break

    # Process file if available
    if (file_path and os.path.exists(file_path)) or st.session_state.current_df is not None:
        if file_path and os.path.exists(file_path):
            st.markdown("""
            <div class="success-message">
                ✅ <strong>File Loaded Successfully!</strong>
            </div>
            """, unsafe_allow_html=True)

        try:
            # Get sheet names
            if file_path and os.path.exists(file_path):
                sheet_names = ExcelHandler.get_sheet_names(file_path)
            else:
                sheet_names = [st.session_state.loaded_sheet] if st.session_state.loaded_sheet else []

            if sheet_names:
                default_index = 0
                if st.session_state.loaded_sheet and st.session_state.loaded_sheet in sheet_names:
                    default_index = sheet_names.index(st.session_state.loaded_sheet)

                selected_sheet = st.selectbox(
                    "📋 Select Sheet:",
                    sheet_names,
                    key="sheet_selector",
                    index=default_index
                )

                # Only reload if needed
                if file_path and os.path.exists(file_path) and (st.session_state.current_df is None or 
                    st.session_state.loaded_sheet != selected_sheet):
                    
                    df = ExcelHandler.get_dataframe_from_sheet(file_path, selected_sheet)
                    df.columns = df.columns.astype(str)

                    # Detect mixed types
                    mixed_type_cols = []
                    column_types = {}

                    for col in df.columns:
                        dtype = pd.api.types.infer_dtype(df[col])
                        column_types[col] = dtype
                        if dtype == 'mixed':
                            mixed_type_cols.append(col)

                    # Display diagnostics
                    with st.expander("🔍 Column Type Diagnostics", expanded=False):
                        st.markdown("**Inferred types for each column:**")
                        type_info = pd.DataFrame({
                            'Column': list(column_types.keys()),
                            'Inferred Type': list(column_types.values())
                        })
                        st.dataframe(type_info, use_container_width=True, hide_index=True)

                        if mixed_type_cols:
                            st.warning(f"⚠️ Found {len(mixed_type_cols)} column(s) with mixed types")
                        else:
                            st.success("✅ No mixed type columns detected!")

                    # Fix mixed types
                    if mixed_type_cols:
                        df[mixed_type_cols] = df[mixed_type_cols].astype(str)
                        st.info(f"🔧 Fixed {len(mixed_type_cols)} column(s)")

                    st.session_state.current_df = df
                    st.session_state.loaded_sheet = selected_sheet
                else:
                    df = st.session_state.current_df

                # Display metrics
                display_data_info(df)

                # Show data preview
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df.head(20), use_container_width=True)

                # Query interface
                st.markdown("---")
                st.subheader("🔍 Query Your Data with Natural Language")

                with st.expander("💡 Example Queries"):
                    st.markdown("""
                    - "Show me the first 10 rows"
                    - "What is the average of column_name?"
                    - "Show rows where value > 100"
                    - "Count unique values in column_name"
                    - "Show summary statistics"
                    """)

                query = st.text_input(
                    "Ask a question about your data:",
                    placeholder="E.g., 'Show top 10 rows with highest values'",
                    key="nlq_input"
                )

                if query:
                    if st.session_state.current_df is None:
                        st.error("❌ No data loaded. Please upload a file first.")
                    else:
                        with st.spinner("🤖 Processing your query..."):
                            try:
                                llm = st.session_state.llm_handler
                                query_df = st.session_state.current_df.copy()

                                df_description = ExcelHandler.get_dataframe_description(query_df)

                                code_response = llm.natural_language_to_pandas(
                                    json.dumps(df_description, indent=2),
                                    query
                                )

                                # Display code
                                with st.expander("📝 Generated Code"):
                                    st.code(code_response, language="python")

                                # Execute code
                                try:
                                    stdout_buffer = StringIO()
                                    stderr_buffer = StringIO()
                                    namespace = {'df': query_df, 'pd': pd, '__builtins__': __builtins__}
                                    code = code_response.strip()

                                    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                                        try:
                                            result = eval(code, namespace)
                                        except SyntaxError:
                                            exec(code, namespace)
                                            if 'result' in namespace:
                                                result = namespace['result']
                                            else:
                                                lines = [line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
                                                if lines:
                                                    try:
                                                        result = eval(lines[-1], namespace)
                                                    except:
                                                        result = namespace.get('df')
                                                else:
                                                    result = namespace.get('df')

                                    stdout_output = stdout_buffer.getvalue()
                                    stderr_output = stderr_buffer.getvalue()
                                    console_output = "\n".join(section for section in [stdout_output.strip(), stderr_output.strip()] if section)

                                    st.success("✅ Query executed successfully!")

                                    if console_output:
                                        with st.expander("📝 Console Output", expanded=True):
                                            st.code(console_output, language="text")

                                    st.subheader("📊 Result:")
                                    if result is not None:
                                        if isinstance(result, pd.DataFrame):
                                            st.dataframe(result, use_container_width=True)
                                        elif isinstance(result, pd.Series):
                                            st.dataframe(result.to_frame(), use_container_width=True)
                                        else:
                                            st.write(result)

                                        # Export options
                                        if isinstance(result, (pd.DataFrame, pd.Series)):
                                            st.markdown("---")
                                            st.subheader("💾 Export Results")
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                @st.cache_data
                                                def convert_to_csv(df):
                                                    return df.to_csv(index=False).encode('utf-8')

                                                csv_data = convert_to_csv(result if isinstance(result, pd.DataFrame) else result.to_frame())
                                                st.download_button(
                                                    label="📥 Download as CSV",
                                                    data=csv_data,
                                                    file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                    mime="text/csv",
                                                    use_container_width=True
                                                )

                                            with col2:
                                                @st.cache_data
                                                def convert_to_excel(df):
                                                    output = BytesIO()
                                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                                        df.to_excel(writer, sheet_name='Results', index=False)
                                                    return output.getvalue()

                                                excel_data = convert_to_excel(result if isinstance(result, pd.DataFrame) else result.to_frame())
                                                st.download_button(
                                                    label="📊 Download as Excel",
                                                    data=excel_data,
                                                    file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                    use_container_width=True
                                                )

                                except Exception as e:
                                    st.error(f"❌ Error executing query: {str(e)}")
                                    st.info("The generated code might need adjustment. Try rephrasing your question.")

                            except Exception as e:
                                error_msg = str(e)
                                if "Error generating pandas code" in error_msg:
                                    st.error(f"❌ Error generating pandas code: {error_msg}")
                                else:
                                    st.error(f"❌ Error: {error_msg}")
                                st.info("💡 **Troubleshooting:**\n1. Make sure Ollama is running\n2. Try rephrasing your question")

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("📤 Please upload an Excel file to begin analysis.")


# ============= DOCUMENT SUMMARIZER MODE =============

with tab_doc:
    
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload Document")
        st.markdown("""
        <div class="upload-area">
            <span>Drag & drop PDF or Word document here</span>
            or click to browse
        </div>
        """, unsafe_allow_html=True)
        uploaded_document = st.file_uploader(
            "Drag & drop PDF/Word file or click to browse",
            type=["pdf", "docx", "doc"],
            key="doc_uploader",
            label_visibility="collapsed"
        )
        st.caption("Supports PDF, DOCX, and DOC files up to 30 MB.")
    
    with col2:
        st.subheader("⚙️ Settings")
        summary_length = st.select_slider(
            "Summary Length:",
            options=["Short (500 words)", "Medium (800 words)", "Long (2000 words)"],
            value="Medium (800 words)"
        )

    if uploaded_document:
        doc_path = StorageManager.save_uploaded_file(uploaded_document, "document")
        st.markdown("""
        <div class="success-message">
            ✅ <strong>Document Loaded Successfully!</strong>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("📖 Extracting text from document..."):
            try:
                text = DocumentHandler.extract_text(doc_path)

                word_count = len(text.split())
                char_count = len(text)

                col1, col2, col3 = st.columns(3)
                file_suffix = Path(uploaded_document.name).suffix.upper().replace('.', '')
                with col1:
                    st.metric("📝 Total Words", f"{word_count:,}")
                with col2:
                    st.metric("📄 Characters", f"{char_count:,}")
                with col3:
                    st.metric("📄 File Type", file_suffix or "Unknown")

                with st.expander("📋 Document Preview"):
                    preview_text = text[:1000] + "..." if len(text) > 1000 else text
                    st.text_area("Text Preview", preview_text, height=200)

                if st.button("✨ Generate Summary", use_container_width=True, type="primary"):
                    with st.spinner("🤖 Generating summary..."):
                        try:
                            llm = st.session_state.llm_handler

                            length_map = {
                                "Short (500 words)": 500,
                                "Medium (800 words)": 800,
                                "Long (2000 words)": 2000,
                            }
                            max_length = length_map.get(summary_length, 2000)

                            summary = llm.summarize_text(text, max_length)
                            summary_path = StorageManager.save_summary(uploaded_document.name, summary)
                            
                            # Store summary in session state for report generation
                            st.session_state.processed_documents[uploaded_document.name] = summary

                            st.success("✅ Summary generated successfully!")
                            st.subheader("📝 Summary:")
                            st.write(summary)

                            st.download_button(
                                label="⬇️ Download Summary",
                                data=summary,
                                file_name=f"{Path(uploaded_document.name).stem}_summary.txt",
                                mime="text/plain",
                                use_container_width=True
                            )

                        except Exception as e:
                            error_msg = str(e)
                            st.error(f"❌ Error generating summary: {error_msg}")
                            st.info("💡 **Troubleshooting:**\n1. Make sure Ollama is running\n2. Check models are installed")

            except Exception as e:
                st.error(f"❌ Error extracting text: {str(e)}")

    else:
        st.info("📤 Please upload a PDF or Word document for processing.")


# ============= REPORT GENERATION TAB =============

with tab_reports:
    st.markdown("### 📑 Automated Report Generation")
    st.markdown("Generate professional reports from your analyzed data")
    
    # Import report generator
    from utils.report_generator import ReportGenerator
    from config import REPORT_DIR
    
    # Report type selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        report_type = st.selectbox(
            "Select Report Type",
            options=["executive", "technical", "business"],
            format_func=lambda x: {
                "executive": "📊 Executive Summary - High-level insights",
                "technical": "🔬 Technical Report - Detailed analysis",
                "business": "💼 Business Intelligence - KPI-focused"
            }[x],
            key="report_type_selector"
        )
    
    st.markdown("---")
    
    # Excel Report Generation
    st.markdown("#### 📊 Generate Excel Data Report")
    
    if st.session_state.current_file and os.path.exists(st.session_state.current_file):
        file_name = Path(st.session_state.current_file).name
        
        if st.button("🚀 Generate Excel Report", key="gen_excel_report", use_container_width=True):
            if st.session_state.current_df is None:
                st.error("❌ No data loaded. Please load an Excel file and sheet in the Excel Analysis tab first.")
            else:
                with st.spinner(f"Generating {report_type} report..."):
                    try:
                        # Initialize report generator
                        report_gen = ReportGenerator(
                            llm_handler=st.session_state.llm_handler,
                            report_dir=REPORT_DIR
                        )
                        
                        # Generate report
                        report_content, report_path = report_gen.generate_excel_report(
                            df=st.session_state.current_df,
                            filename=file_name,
                            report_type=report_type
                        )
                        
                        st.success(f"✅ Report generated successfully!")
                        
                        # Display report
                        st.markdown("### 📄 Generated Report Preview")
                        st.markdown(report_content)
                        
                        # Download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📥 Download Report (Markdown)",
                                data=report_content,
                                file_name=f"{report_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Convert to HTML for better viewing
                            html_content = f"""
                            <html>
                            <head>
                                <style>
                                    body {{ font-family: Arial, sans-serif; padding: 20px; }}
                                    table {{ border-collapse: collapse; width: 100%; }}
                                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                                    th {{ background-color: #4CAF50; color: white; }}
                                </style>
                            </head>
                            <body>
                                <pre>{report_content}</pre>
                            </body>
                            </html>
                            """
                            st.download_button(
                                label="📥 Download Report (HTML)",
                                data=html_content,
                                file_name=f"{report_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                mime="text/html",
                                use_container_width=True
                            )
                        
                        st.info(f"💾 Report saved to: `{report_path}`")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
                        logger.error(f"Report generation error: {e}")
    else:
        st.info("📤 Please upload and load an Excel file first in the Excel Analysis tab")
    
    st.markdown("---")
    
    # Document Report Generation
    st.markdown("#### 📄 Generate Document Summary Report")
    
    if st.session_state.processed_documents:
        selected_doc = st.selectbox(
            "Select document to generate report",
            options=list(st.session_state.processed_documents.keys()),
            key="report_doc_select"
        )
        
        if st.button("🚀 Generate Document Report", key="gen_doc_report", use_container_width=True):
            with st.spinner(f"Generating {report_type} document report..."):
                try:
                    # Get the summary
                    summary = st.session_state.processed_documents[selected_doc]
                    
                    # Initialize report generator
                    report_gen = ReportGenerator(
                        llm_handler=st.session_state.llm_handler,
                        report_dir=REPORT_DIR
                    )
                    
                    # Generate report
                    report_content, report_path = report_gen.generate_document_report(
                        summary=summary,
                        filename=selected_doc,
                        report_type=report_type
                    )
                    
                    st.success(f"✅ Document report generated successfully!")
                    
                    # Display report
                    st.markdown("### 📄 Generated Report Preview")
                    st.markdown(report_content)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Document Report",
                        data=report_content,
                        file_name=f"doc_{report_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
                    st.info(f"💾 Report saved to: `{report_path}`")
                    
                except Exception as e:
                    st.error(f"❌ Error generating document report: {str(e)}")
                    logger.error(f"Document report generation error: {e}")
    else:
        st.info("📤 Please process a document first in the Document Summarizer tab")


# ============= FOOTER =============

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    Built with ❤️ using Streamlit • Advanced AI-Powered Data Analysis & Document Intelligence<br>
    <em>Upload your data, ask questions, get intelligent insights!</em>
</div>
""", unsafe_allow_html=True)