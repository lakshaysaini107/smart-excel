import ollama
import chromadb
import streamlit as st
import pandas as pd

class OllamaRAGSystem:
    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name
        self.embedding_model = "nomic-embed-text"
        
        # Local ChromaDB storage
        self.client = chromadb.PersistentClient(path="C:\\Users\\laksh\\OneDrive\\Desktop\\smart excel chroma vdb")
        self.collection = self.client.get_or_create_collection("excel_analytics")
    
    def process_excel_data(self, df):
        """Convert Excel data to embeddings with minimal changes"""
        chunks = []
        for idx, row in df.iterrows():
            # Create meaningful text representation
            text = f"Row {idx}: " + ", ".join([f"{col}: {val}" for col, val in row.items()])
            chunks.append({"id": f"row_{idx}", "text": text, "metadata": {"row": idx}})
        
        # Store in local vector database
        self.add_to_vector_store(chunks)
        return len(chunks)
    
    def add_to_vector_store(self, chunks):
        """Add chunks to ChromaDB with Ollama embeddings"""
        for chunk in chunks:
            # Generate embedding locally with Ollama
            embedding = ollama.embeddings(model=self.embedding_model, prompt=chunk["text"])
            
            self.collection.add(
                embeddings=[embedding["embedding"]],
                documents=[chunk["text"]],
                metadatas=[chunk["metadata"]],
                ids=[chunk["id"]]
            )
    
    def query(self, question):
        """Query with natural language using local LLM"""
        # Get question embedding
        q_embedding = ollama.embeddings(model=self.embedding_model, prompt=question)
        
        # Search local vector database
        results = self.collection.query(
            query_embeddings=[q_embedding["embedding"]], 
            n_results=5
        )
        
        # Generate answer with local LLM
        context = "\n".join(results["documents"][0])
        prompt = f"""Based on this Excel data:
{context}

Question: {question}
Answer with precise information from the data:"""
        
        response = ollama.generate(
            model=self.model_name, 
            prompt=prompt,
            options={"temperature": 0.2}  # Low temperature for accuracy
        )
        return response["response"]
