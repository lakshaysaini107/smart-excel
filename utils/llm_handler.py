try:
    from langchain_ollama import OllamaLLM, OllamaEmbeddings
except Exception:  # pragma: no cover - optional dependency may not be installed in every env
    OllamaLLM = None
    OllamaEmbeddings = None
# Text splitter: prefer the standard `langchain.text_splitter` if available,
# otherwise fall back to local shim `langchain_text_splitters`.
try:
    from langchain.text_splitter import CharacterTextSplitter  # type: ignore
except Exception:
    try:
        from langchain_text_splitters import CharacterTextSplitter  # type: ignore
    except Exception:
        # minimal fallback
        class CharacterTextSplitter:
            def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_text(self, text: str):
                return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

# Vector store: try community package first, then fallback to stub
try:
    from langchain_community.vectorstores import Chroma
except Exception:
    try:
        from langchain_community.vectorstores import Chroma  # type: ignore
    except Exception:
        class Chroma:
            def __init__(self, persist_directory: str = None, embedding_function=None):
                self.persist_directory = persist_directory
                self.embedding_function = embedding_function

            def add_documents(self, docs):
                raise RuntimeError("Chroma stub: install real package to use")

# RetrievalQA: prefer `langchain.chains`, fallback to `langchain_core.chains` stub
try:
    from langchain.chains import RetrievalQA  # type: ignore
except Exception:
    try:
        from langchain_core.chains import RetrievalQA  # type: ignore
    except Exception:
        class RetrievalQA:
            def __init__(self, llm=None, retriever=None, **kwargs):
                self.llm = llm
                self.retriever = retriever

            def run(self, query: str):
                raise RuntimeError("RetrievalQA stub: install real package to use")
from config import (
    LLM_MODEL, EMBEDDING_MODEL, OLLAMA_HOST,
    CHROMA_PERSIST_DIR, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP
)

class LLMHandler:
    """Handles all LLM operations through Ollama"""
    
    def __init__(self):
        """Initialize LLM and Embeddings"""
        if OllamaLLM is None or OllamaEmbeddings is None:
            raise Exception("langchain_ollama package is not installed. Please install it with: pip install langchain-ollama")
        
        try:
            # Initialize LLM
            self.llm = OllamaLLM(
                model=LLM_MODEL,
                base_url=OLLAMA_HOST
            )
            
            # Initialize Embeddings
            self.embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OLLAMA_HOST
            )
        
        except Exception as e:
            raise Exception(f"Error initializing LLM: {str(e)}")
    
    def summarize_text(self, text, max_length=150):
        """Summarize text using local LLM"""
        try:
            # Truncate input if too large
            if len(text) > 4000:
                text = text[:4000]
            
            prompt = f"""You are a professional summarizer. Summarize the following text concisely in approximately {max_length} words. Focus on key points and main ideas.

TEXT TO SUMMARIZE:
{text}

SUMMARY:"""
            
            response = self.llm.invoke(prompt)
            return response.strip()
        
        except Exception as e:
            raise Exception(f"Error summarizing text: {str(e)}")
    
    def natural_language_to_pandas(self, df_schema, query):
        """Convert natural language query to pandas code"""
        try:
            prompt = f"""You are a Python expert specializing in pandas data manipulation.
Convert the following natural language question into valid pandas code.
Return ONLY the Python code that would execute on a DataFrame named 'df'.
Do not include explanations, markdown, or code blocks - just the raw Python code.

DataFrame Schema and Info:
{df_schema}

Question: {query}

Return only valid Python code:"""
            
            response = self.llm.invoke(prompt)
            
            # Clean up the response
            code = response.strip()
            
            # Remove markdown code blocks if present
            if "```" in code:
                parts = code.split("```")
                content = parts[1] if len(parts) > 1 else code
                if content.lstrip().startswith("python"):
                    content = content.lstrip()[6:]
                    # drop a leading newline if present after the language tag
                    if content.startswith("\n"):
                        content = content[1:]
                code = content.strip()
            
            return code.strip()
        
        except Exception as e:
            raise Exception(f"Error generating pandas code: {str(e)}")
