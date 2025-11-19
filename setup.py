"""
Setup script to verify installation and check dependencies.
Run this before starting the app to ensure everything is configured correctly.
"""
import sys
import subprocess

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name.replace("-", "_")
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        return False

def check_ollama_running():
    """Check if Ollama server is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama server is running")
            return True
        else:
            print("⚠️  Ollama server responded but with unexpected status")
            return False
    except Exception as e:
        print(f"❌ Ollama server is NOT running: {str(e)}")
        print("   Start it with: ollama serve")
        return False

def main():
    print("=" * 60)
    print("Data Analysis & Document Summarizer - Setup Check")
    print("=" * 60)
    print()
    
    print("Checking Python packages...")
    print("-" * 60)
    
    packages = [
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("openpyxl", "openpyxl"),
        ("langchain-ollama", "langchain_ollama"),
        ("langchain-community", "langchain_community"),
        ("langchain-core", "langchain_core"),
        ("chromadb", "chromadb"),
        ("pymupdf", "fitz"),  # pymupdf imports as fitz
        ("PyPDF2", "PyPDF2"),
    ]
    
    missing = []
    for package, import_name in packages:
        if not check_package(package, import_name):
            missing.append(package)
    
    print()
    print("Checking Ollama connection...")
    print("-" * 60)
    ollama_ok = check_ollama_running()
    
    print()
    print("=" * 60)
    
    if missing:
        print("❌ Missing packages detected!")
        print("\nInstall missing packages with:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr install all dependencies:")
        print("  pip install -r requirements.txt")
        return 1
    
    if not ollama_ok:
        print("⚠️  Ollama server is not running")
        print("   Start it with: ollama serve")
        print("   Then download models: ollama pull llama3.2")
        return 1
    
    print("✅ All checks passed! You're ready to run the app.")
    print("\nStart the app with:")
    print("  streamlit run app.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())

