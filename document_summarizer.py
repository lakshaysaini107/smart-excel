from typing import List, Dict, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import networkx as nx
from collections import Counter
import re

class AdvancedDocumentSummarizer:
    """Advanced document summarization with multiple approaches"""
    
    def __init__(self):
        # Initialize summarization pipeline
        try:
            self.abstractive_summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Could not load BART model: {e}")
            self.abstractive_summarizer = None
            
        # Initialize sentence embeddings for extractive summarization
        try:
            self.sentence_model = pipeline(
                "feature-extraction",
                model="sentence-transformers/all-MiniLM-L6-v2",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_model = None

    def summarize_documents(self, documents: List[str],
                           summary_type: str = "hybrid",
                           max_length: int = 150,
                           query_focused: str = None) -> Dict:
        """
        Multi-approach document summarization
        
        Args:
            documents: List of document texts
            summary_type: "extractive", "abstractive", or "hybrid"
            max_length: Maximum length of summary
            query_focused: Optional query to focus summarization
        """
        try:
            if not documents:
                return {'error': 'No documents provided'}
                
            combined_text = " ".join(documents)
            
            results = {
                'document_count': len(documents),
                'total_length': len(combined_text),
                'summaries': {},
                'key_insights': [],
                'query_relevance': None
            }
            
            # Extractive summarization
            if summary_type in ["extractive", "hybrid"]:
                extractive_summary = self._extractive_summarization(
                    documents, max_length, query_focused
                )
                results['summaries']['extractive'] = extractive_summary
            
            # Abstractive summarization
            if summary_type in ["abstractive", "hybrid"] and self.abstractive_summarizer:
                abstractive_summary = self._abstractive_summarization(
                    combined_text, max_length
                )
                results['summaries']['abstractive'] = abstractive_summary
            
            # Key insights extraction
            results['key_insights'] = self._extract_key_insights(combined_text)
            
            # Query relevance scoring
            if query_focused:
                results['query_relevance'] = self._calculate_query_relevance(
                    combined_text, query_focused
                )
            
            # Generate final hybrid summary
            if summary_type == "hybrid":
                results['summaries']['hybrid'] = self._create_hybrid_summary(
                    results['summaries'], max_length
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Document summarization failed: {e}")
            return {'error': str(e)}

    def _extractive_summarization(self, documents: List[str],
                                 max_length: int,
                                 query_focused: str = None) -> Dict:
        """
        Extractive summarization using sentence scoring
        """
        try:
            # Combine and split into sentences
            all_sentences = []
            doc_sources = []
            
            for i, doc in enumerate(documents):
                sentences = self._split_sentences(doc)
                all_sentences.extend(sentences)
                doc_sources.extend([i] * len(sentences))
            
            if not all_sentences:
                return {'summary': '', 'selected_sentences': [], 'method': 'extractive'}
            
            # Score sentences using multiple criteria
            sentence_scores = {}
            for i, sentence in enumerate(all_sentences):
                score = 0
                
                # TF-IDF based scoring
                score += self._calculate_tfidf_score(sentence, all_sentences)
                
                # Position-based scoring (earlier sentences often more important)
                position_weight = 1 - (i / len(all_sentences)) * 0.3
                score *= position_weight
                
                # Length-based scoring (moderate length preferred)
                length_score = self._calculate_length_score(sentence)
                score *= length_score
                
                # Query relevance scoring
                if query_focused:
                    query_relevance = self._calculate_sentence_query_relevance(
                        sentence, query_focused
                    )
                    score *= (1 + query_relevance)
                
                sentence_scores[i] = score
            
            # Select top sentences
            num_sentences = min(
                max(1, max_length // 50),  # Approximate sentences for length
                len(all_sentences) // 3   # Max 1/3 of original sentences
            )
            
            top_sentence_indices = sorted(
                sentence_scores.keys(),
                key=lambda x: sentence_scores[x],
                reverse=True
            )[:num_sentences]
            
            # Sort by original order for coherent summary
            top_sentence_indices.sort()
            selected_sentences = [all_sentences[i] for i in top_sentence_indices]
            summary_text = " ".join(selected_sentences)
            
            return {
                'summary': summary_text[:max_length],
                'selected_sentences': selected_sentences,
                'sentence_scores': {i: sentence_scores[i] for i in top_sentence_indices},
                'method': 'extractive'
            }
            
        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            return {'summary': '', 'error': str(e), 'method': 'extractive'}

    def _abstractive_summarization(self, text: str, max_length: int) -> Dict:
        """
        Abstractive summarization using transformer model
        """
        try:
            if not self.abstractive_summarizer:
                return {'summary': '', 'error': 'Abstractive model not available', 'method': 'abstractive'}
            
            # Split long texts into chunks
            max_input_length = 1024  # BART input limit
            text_chunks = self._split_text_for_model(text, max_input_length)
            chunk_summaries = []
            
            for chunk in text_chunks:
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                    
                try:
                    summary_result = self.abstractive_summarizer(
                        chunk,
                        max_length=min(max_length, 150),
                        min_length=max(1, max_length // 4),
                        do_sample=False,
                        truncation=True
                    )
                    chunk_summaries.append(summary_result['summary_text'])
                except Exception as chunk_error:
                    logger.warning(f"Failed to summarize chunk: {chunk_error}")
                    continue
            
            # Combine chunk summaries
            if chunk_summaries:
                if len(chunk_summaries) == 1:
                    final_summary = chunk_summaries[0]
                else:
                    # Recursively summarize if multiple chunks
                    combined_summaries = " ".join(chunk_summaries)
                    if len(combined_summaries) > max_input_length:
                        # Need to summarize the summaries
                        final_result = self.abstractive_summarizer(
                            combined_summaries,
                            max_length=max_length,
                            min_length=max_length // 4,
                            do_sample=False,
                            truncation=True
                        )
                        final_summary = final_result['summary_text']
                    else:
                        final_summary = combined_summaries
            else:
                final_summary = text[:max_length]  # Fallback to truncation
            
            return {
                'summary': final_summary,
                'chunks_processed': len(text_chunks),
                'chunk_summaries': chunk_summaries,
                'method': 'abstractive'
            }
            
        except Exception as e:
            logger.error(f"Abstractive summarization failed: {e}")
            return {'summary': '', 'error': str(e), 'method': 'abstractive'}

    def _create_hybrid_summary(self, summaries: Dict, max_length: int) -> Dict:
        """
        Create hybrid summary combining extractive and abstractive approaches
        """
        try:
            extractive_summary = summaries.get('extractive', {}).get('summary', '')
            abstractive_summary = summaries.get('abstractive', {}).get('summary', '')
            
            if not extractive_summary and not abstractive_summary:
                return {'summary': '', 'method': 'hybrid', 'error': 'No summaries available'}
            
            if not extractive_summary:
                return {'summary': abstractive_summary, 'method': 'hybrid', 'source': 'abstractive_only'}
            
            if not abstractive_summary:
                return {'summary': extractive_summary, 'method': 'hybrid', 'source': 'extractive_only'}
            
            # Combine summaries intelligently
            # Use extractive for key facts, abstractive for fluency
            # Calculate similarity to avoid redundancy
            similarity = self._calculate_text_similarity(extractive_summary, abstractive_summary)
            
            if similarity > 0.7:
                # High similarity - choose the more fluent (abstractive)
                final_summary = abstractive_summary
                combination_method = 'abstractive_selected_high_similarity'
            else:
                # Low similarity - combine both
                combined = f"{abstractive_summary} Key details: {extractive_summary}"
                
                # Truncate to max length
                if len(combined) > max_length:
                    # Prioritize abstractive, add key extractive points
                    available_space = max_length - len(abstractive_summary) - 20
                    if available_space > 50:
                        key_extractive = extractive_summary[:available_space]
                        final_summary = f"{abstractive_summary} Key details: {key_extractive}..."
                    else:
                        final_summary = abstractive_summary
                else:
                    final_summary = combined
                    
                combination_method = 'combined_low_similarity'
            
            return {
                'summary': final_summary,
                'method': 'hybrid',
                'combination_method': combination_method,
                'similarity_score': similarity
            }
            
        except Exception as e:
            logger.error(f"Hybrid summary creation failed: {e}")
            return {'summary': '', 'error': str(e), 'method': 'hybrid'}

    def _extract_key_insights(self, text: str) -> List[Dict]:
        """
        Extract key insights and important points from text
        """
        try:
            insights = []
            
            # Extract numerical insights
            number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:%|\s*percent)?\b'
            numbers = re.findall(number_pattern, text)
            if numbers:
                insights.append({
                    'type': 'numerical',
                    'description': f"Contains {len(numbers)} numerical values",
                    'values': numbers[:10]  # Limit to prevent spam
                })
            
            # Extract trend indicators
            trend_keywords = ['increase', 'decrease', 'growth', 'decline', 'rise', 'fall',
                            'improve', 'worsen', 'better', 'worse', 'higher', 'lower']
            trend_mentions = []
            for keyword in trend_keywords:
                if keyword in text.lower():
                    # Find sentences containing trend keywords
                    sentences = self._split_sentences(text)
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            trend_mentions.append(sentence)
                            break
            
            if trend_mentions:
                insights.append({
                    'type': 'trends',
                    'description': f"Identified {len(trend_mentions)} trend indicators",
                    'examples': trend_mentions[:3]
                })
            
            # Extract entities (simplified)
            entity_patterns = {
                'dates': r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b',
                'percentages': r'\b\d+(?:\.\d+)?%\b',
                'currencies': r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
            }
            
            for entity_type, pattern in entity_patterns.items():
                matches = re.findall(pattern, text)
                if matches:
                    insights.append({
                        'type': entity_type,
                        'description': f"Found {len(matches)} {entity_type}",
                        'examples': matches[:5]
                    })
            
            return insights
            
        except Exception as e:
            logger.error(f"Key insights extraction failed: {e}")
            return []

    # Helper methods for the summarizer
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be enhanced with NLTK)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _calculate_tfidf_score(self, sentence: str, all_sentences: List[str]) -> float:
        """Calculate TF-IDF based sentence importance score"""
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(all_sentences)
            
            # Find sentence index
            sentence_idx = all_sentences.index(sentence)
            sentence_tfidf = tfidf_matrix[sentence_idx].toarray()
            
            # Return sum of TF-IDF scores (higher = more important)
            return sentence_tfidf.sum()
        except Exception:
            return 1.0  # Default score

    def _calculate_length_score(self, sentence: str) -> float:
        """Score based on sentence length (moderate length preferred)"""
        length = len(sentence.split())
        if length < 5:
            return 0.5  # Too short
        elif 5 <= length <= 25:
            return 1.0  # Ideal length
        elif 25 < length <= 40:
            return 0.8  # Acceptable
        else:
            return 0.6  # Too long

    def _calculate_sentence_query_relevance(self, sentence: str, query: str) -> float:
        """Calculate relevance of sentence to query"""
        try:
            sentence_words = set(sentence.lower().split())
            query_words = set(query.lower().split())
            if not query_words:
                return 0
            
            # Jaccard similarity
            intersection = sentence_words.intersection(query_words)
            union = sentence_words.union(query_words)
            return len(intersection) / len(union) if union else 0
        except Exception:
            return 0

    def _split_text_for_model(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks suitable for model input"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > max_length:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0
        except Exception:
            return 0

    def _calculate_query_relevance(self, text: str, query: str) -> float:
        """Calculate overall relevance of text to query"""
        try:
            # Simple keyword-based relevance
            text_words = text.lower().split()
            query_words = query.lower().split()
            matches = sum(1 for word in query_words if word in text_words)
            return matches / len(query_words) if query_words else 0
        except Exception:
            return 0