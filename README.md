# Data Analysis & Document Summarizer

A Streamlit application powered by local Ollama LLM (Llama 3.2) for Excel data analysis and document summarization.

## Features

- 📊 **Excel Data Analysis**: Upload Excel files and query them using natural language
- 📄 **Document Summarization**: Summarize PDF and text documents
- 🤖 **Local LLM**: Uses Ollama for privacy-focused AI processing
- 🔍 **Mixed Type Detection**: Automatically detects and fixes columns with mixed data types

## Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running
   - Download from: https://ollama.com/download
   - Start the server: `ollama serve`

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and pull required Ollama models**
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

## Configuration

Edit `config.py` to customize:
- Model names (`LLM_MODEL`, `EMBEDDING_MODEL`)
- Ollama host URL (`OLLAMA_HOST`)
- Directory paths (automatically created if they don't exist)

## Running the Application

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

### Excel Analysis Mode

1. Upload an Excel file (.xlsx or .xls)
2. Select a sheet from the dropdown
3. The app will automatically detect and fix mixed data type columns
4. Ask questions about your data in natural language
   - Example: "Show me the first 10 rows"
   - Example: "What is the average of column_name?"
   - Example: "Show rows where value > 100"

### Document Summarizer Mode

1. Upload a PDF or text file
2. Select summary length (Short/Medium/Long)
3. Click "Generate Summary"
4. Download the summary if needed

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── utils/
│   ├── excel_handler.py   # Excel file processing
│   ├── document_handler.py # Document text extraction
│   ├── llm_handler.py     # LLM operations
│   └── storage_manager.py # File storage management
├── uploaded_files/        # Uploaded Excel/Document files (auto-created)
├── vector_db/            # ChromaDB vector database (auto-created)
└── summaries/            # Generated summaries (auto-created)
```

## Troubleshooting

### Error: "langchain_ollama package is not installed"
- Solution: `pip install langchain-ollama`

### Error: "Could not connect to Ollama"
- Solution: Make sure Ollama is running: `ollama serve`
- Check that Ollama is accessible at `http://localhost:11434`

### Error: "Model not found"
- Solution: Download the required models:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

### Mixed Type Columns
- The app automatically detects and converts mixed type columns to strings
- Check the "Column Type Diagnostics" section to see inferred types

## Deployment

### Local Deployment
The app runs locally using Streamlit. No cloud services required.

### Production Deployment (Streamlit Cloud)
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Set environment variables if needed
4. Note: Ollama must be running on the deployment server

### Docker Deployment (Optional)
Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## License

This project is open source and available for personal and commercial use.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed
3. Verify Ollama is running and models are downloaded

