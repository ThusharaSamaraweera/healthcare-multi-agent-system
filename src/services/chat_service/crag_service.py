from typing import Any, Dict
from loguru import logger
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from infrastructure.config import TOP_K_RESULTS, CRAG_EXPANDED_K, CRAG_CONFIDENCE_THRESHOLD
from infrastructure.utils import calculate_confidence, format_docs

from services.chat_service.rag_templates import RAG_TEMPLATE

class CRAGService:
    
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Any,
        initial_k: int = TOP_K_RESULTS,
        expanded_k: int = CRAG_EXPANDED_K
    ):
        """
        Initialize CRAG service.
        """
        self.retriever = retriever
        self.llm = llm
        self.initial_k = initial_k
        self.expanded_k = expanded_k
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    def generate(
            self,
            query: str,
            confidence_threshold: float = CRAG_CONFIDENCE_THRESHOLD,
            verbose: bool = True
        ) -> Dict[str, Any]:
            """
            Generate answer with CRAG (Corrective RAG).
            """
            if verbose:
                logger.info(f"🔍 Query: {query}")
                logger.success(f"🎯 Confidence threshold: {confidence_threshold}\n")
            
            # Step 1: Initial retrieval
            if verbose:
                logger.info(f"1️⃣  Initial retrieval (k={self.initial_k})...")
            
            self._set_k(self.initial_k)
            docs_initial = self.retriever.invoke(query)
            confidence_initial = calculate_confidence(docs_initial, query)
            
            if verbose:
                logger.info(f"   📊 Confidence: {confidence_initial:.2f}")
            
            # Step 2: Check if correction needed
            if confidence_initial >= confidence_threshold:
                if verbose:
                    logger.success(f"   ✅ Confidence sufficient - proceeding with initial retrieval")
                final_docs = docs_initial
                confidence_final = confidence_initial
                correction_applied = False
            else:
                if verbose:
                    logger.warning(f"   ⚠️  Low confidence - applying corrective retrieval...\n")
                
                # Step 3: Corrective retrieval
                if verbose:
                    logger.info(f"2️⃣  Corrective retrieval (k={self.expanded_k}, expanded)...")
                
                # Expand k for more diverse results
                self._set_k(self.expanded_k)
                docs_corrected = self.retriever.invoke(query)
                confidence_final = calculate_confidence(docs_corrected, query)
                
                if verbose:
                    logger.info(f"   📊 Corrected confidence: {confidence_final:.2f}")
                    improvement = (confidence_final - confidence_initial) * 100
                    logger.info(f"   📈 Confidence improved by {improvement:.1f}%")
                
                final_docs = docs_corrected
                correction_applied = True
            
            # Step 4: Generate answer
            if verbose:
                logger.info(f"\n3️⃣  Generating answer...")
            
            start = time.time()
            
            # Format docs and generate
            context = format_docs(final_docs)
            prompt_input = {"context": context, "question": query}
            answer = (self.prompt | self.llm | StrOutputParser()).invoke(prompt_input)
            
            elapsed = time.time() - start
            
            # Extract evidence URLs
            evidence_urls = list(set([doc.metadata['url'] for doc in final_docs]))
            
            return {
                'answer': answer,
                'confidence_initial': confidence_initial,
                'confidence_final': confidence_final,
                'correction_applied': correction_applied,
                'docs_used': len(final_docs),
                'generation_time': elapsed,
                'evidence_urls': evidence_urls,
                'evidence': final_docs
            }
        
    def _set_k(self, k: int) -> None:
        """Set retrieval count on the retriever."""
        self.retriever.top_k = k
    