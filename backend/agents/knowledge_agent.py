"""
Agricultural Knowledge AI Agent (Week 6).
Implements Retrieval-Augmented Generation (RAG) over ICAR and KVK guideline manuals.
Retrieves vector-matched pest, disease, and sowing protocols and delivers citation-backed advice.
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_core.tools import tool, BaseTool
from backend.schemas.rag_schemas import (
    KnowledgeQueryRequest,
    KnowledgeAgentResponse,
    KnowledgeSourceItem,
)
from backend.rag.vector_store import get_vector_store

load_dotenv()


def create_knowledge_tool() -> BaseTool:
    """Factory function creating LangChain tool for agronomic RAG search."""
    store = get_vector_store()

    @tool("search_icar_knowledge_tool", args_schema=KnowledgeQueryRequest)
    def search_icar_knowledge_tool(query: str) -> List[Dict[str, Any]]:
        """
        Retrieves official ICAR & KVK agricultural advisory guidelines on pest management,
        fungal diseases, sowing windows, seed treatments, and crop care.
        """
        return store.search(query=query, top_k=2)

    return search_icar_knowledge_tool


class KnowledgeAgent:
    """RAG-powered conversational agent synthesizing agronomic literature into farmer advice."""

    def __init__(self, llm: Optional[Any] = None):
        """Initialize agent with FAISS vector store and optional LLM."""
        self.tool = create_knowledge_tool()
        self.store = get_vector_store()
        self.llm = llm or self._initialize_llm()

    def _initialize_llm(self) -> Optional[Any]:
        """Initialize Groq or OpenAI LLM if configured."""
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key and groq_api_key != "your_groq_api_key_here":
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    api_key=groq_api_key,
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.1
                )
            except Exception:
                pass

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and openai_api_key != "your_openai_api_key_here":
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    api_key=openai_api_key,
                    model="gpt-4o-mini",
                    temperature=0.1
                )
            except Exception:
                pass

        return None

    def run(self, query: str) -> KnowledgeAgentResponse:
        """
        Search ICAR/KVK vector database and synthesize an actionable, cited answer.
        """
        matches = self.tool.invoke({"query": query})

        if not matches:
            return KnowledgeAgentResponse(
                query=query,
                answer="No specific ICAR guidelines found for this query. Please consult your local Krishi Vigyan Kendra (KVK).",
                citations=[],
                raw_context=""
            )

        context_blocks = []
        citations: List[KnowledgeSourceItem] = []

        for m in matches:
            context_blocks.append(f"[{m['section']}]\n{m['content']}")
            citations.append(KnowledgeSourceItem(
                section=m["section"],
                source=m["source"],
                score=float(m["score"])
            ))

        combined_context = "\n\n".join(context_blocks)

        # Synthesize answer using LLM if available
        if self.llm is not None:
            try:
                from langchain_core.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert ICAR/KVK Agricultural Advisor. Answer the farmer's question using ONLY the provided guideline context. Include precise dosage, timings, and safety measures."),
                    ("human", "Question: {query}\n\nGuidelines Context:\n{context}")
                ])
                chain = prompt | self.llm
                llm_res = chain.invoke({"query": query, "context": combined_context})
                answer_text = str(llm_res.content)
                return KnowledgeAgentResponse(
                    query=query,
                    answer=answer_text,
                    citations=citations,
                    raw_context=combined_context
                )
            except Exception:
                pass

        # Deterministic grounded extraction
        top_match = matches[0]
        answer_text = (
            f"🌾 **ICAR / KVK Recommended Practice for '{query}'**\n\n"
            f"{top_match['content']}\n\n"
            f"📌 **Citation:** {top_match['section']} ({top_match['source']})"
        )

        return KnowledgeAgentResponse(
            query=query,
            answer=answer_text,
            citations=citations,
            raw_context=combined_context
        )
