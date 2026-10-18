from typing import Any, Dict, List, Optional
import time

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from infrastructure.utils import format_docs
from infrastructure.db.qdrant_client import search_chunks
from infrastructure.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD
from services.chat_service.rag_templates import RAG_TEMPLATE

class QdrantRetriever(BaseRetriever):
    """
    LangChain-compatible retriever backed by Qdrant Cloud.
    """

    embedder: Any = None
    top_k: int = TOP_K_RESULTS
    score_threshold: float = SIMILARITY_THRESHOLD


    def _get_relevant_documents(
        self,
        query: str,
    ) -> List[Document]:
        """Retrieve relevant documents from Qdrant."""
        # Embed
        query_vec = self.embedder.embed_query(query)

        # Search Qdrant
        hits = search_chunks(
            query_vector=query_vec,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )

        # Convert to LangChain Documents
        # For parent-child: use parent_text as page_content (richer context for LLM)
        # while keeping the child text in metadata for reference.
        seen_parents: set = set()
        docs = []
        for hit in hits:
            parent_text = hit.get("parent_text")
            parent_id = hit.get("parent_id")

            # Deduplicate by parent_id so the LLM doesn't see the same
            # parent passage multiple times when several children match.
            if parent_id and parent_id in seen_parents:
                continue
            if parent_id:
                seen_parents.add(parent_id)

            page_content = parent_text if parent_text else hit["chunk_text"]
            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "url": hit.get("url", ""),
                        "title": hit.get("title", ""),
                        "strategy": hit.get("strategy", ""),
                        "chunk_index": hit.get("chunk_index", 0),
                        "score": hit.get("score", 0.0),
                        "child_text": hit["chunk_text"],
                    },
                )
            )
        return docs



class RAGService:
    """Retrieve relevant knowledge-base chunks for a user query."""

    def __init__(self,
        embedder: Any,
        llm: Any,
        k: int = TOP_K_RESULTS,
        score_threshold: float = SIMILARITY_THRESHOLD,):

        self.embedder = embedder
        self.llm = llm
        self.k = k

        # Build Qdrant retriever
        self.retriever = QdrantRetriever(
            embedder=embedder,
            top_k=k,
            score_threshold=score_threshold,
        )

        # Build LCEL chain
        self.chain = build_rag_chain(self.retriever, llm, k)


    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 4,
        score_threshold: float = 0.0,
        strategy_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Embed *query* and return matching Qdrant chunks."""
        if not query.strip():
            return []

        query_vector = self.embedder.embed_query(query)
        return search_chunks(
            query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            strategy_filter=strategy_filter,
        )


    def generate(self, query: str) -> Dict[str, Any]:
        """
        Generate answer for query using RAG.
        """
        start = time.time()

        # Generate answer
        result = self.chain.invoke(query)

        elapsed = time.time() - start

        # Extract unique URLs
        evidence = result.get("documents", [])
        evidence_urls = list(set(doc.metadata.get("url", "") for doc in evidence))

        return {
            "answer": result.get("answer", ""),
            "evidence": evidence,
            "evidence_urls": evidence_urls,
            "generation_time": elapsed,
            "num_docs": len(evidence),
        }


def build_rag_chain(
    retriever: BaseRetriever,
    llm: Any,
    k: int = TOP_K_RESULTS,
    template: str = RAG_TEMPLATE,
) -> Runnable:
    """
    Build modern RAG chain using LangChain Expression Language (LCEL).
    """

    # Update retriever k if specified
    if hasattr(retriever, "search_kwargs"):
        retriever.search_kwargs["k"] = k

    # Create prompt template (Runnable)
    rag_prompt = ChatPromptTemplate.from_template(template)

    # BUILD THE CHAIN (Modern LCEL approach!)
    def retrieve_once(question: str) -> dict:
        documents = retriever.invoke(question)

        return {
            "question": question,
            "documents": documents,
            "context": format_docs(documents),
        }


    prepared = RunnableLambda(retrieve_once)

    answer_chain = (
        RunnableLambda(
            lambda data: {
                "context": data["context"],
                "question": data["question"],
            }
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = prepared.assign(answer=answer_chain)
    return rag_chain

