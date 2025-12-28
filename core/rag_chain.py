"""
Advanced RAG Chain with Memory and Multi-Provider Support
Includes retry logic for rate limits
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, List, Optional
import logging
import time

from core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class RAGChain:
    """Advanced RAG chain with memory and context awareness"""
    
    def __init__(
        self,
        vectorstore,
        provider: str = "google",
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        k: int = 4,
        memory_k: int = 5
    ):
        """
        Initialize RAG chain
        
        Args:
            vectorstore: Vector store for retrieval
            provider: LLM provider ('google' or 'openai')
            model_name: Model name
            api_key: API key
            temperature: LLM temperature
            k: Number of documents to retrieve
            memory_k: Number of previous conversations to remember
        """
        self.vectorstore = vectorstore
        self.provider = provider
        self.model_name = model_name
        self.k = k
        
        # Initialize memory
        self.memory = MemoryManager(k=memory_k, memory_type="window")
        
        # Initialize LLM
        self.llm = self._initialize_llm(provider, model_name, api_key, temperature)
        
        # Initialize retriever
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Create chain
        self.chain = self._create_chain()
        
        logger.info(f"✅ RAG Chain initialized: {provider}/{model_name}")
    
    def _initialize_llm(self, provider: str, model_name: str, api_key: str, temperature: float):
        """Initialize LLM based on provider"""
        logger.info(f"Initializing {provider} LLM: {model_name}")
        
        if provider == "google":
            # Use simpler model names - the API adds "models/" prefix automatically
            # Don't add "models/" prefix - langchain handles this
            logger.info(f"Using Google model: {model_name}")
            
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=api_key,
                convert_system_message_to_human=True
            )
        elif provider == "openai":
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _format_docs(self, docs) -> str:
        """Format retrieved documents"""
        if not docs:
            return "No relevant documents found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content
            
            formatted.append(
                f"[Document {i}] Source: {source}, Page: {page}\n{content}"
            )
        
        return "\n\n---\n\n".join(formatted)
    
    def _format_chat_history(self, history: List[Dict]) -> str:
        """Format chat history for context"""
        if not history:
            return ""
        
        formatted = []
        for interaction in history:
            formatted.append(f"User: {interaction['question']}")
            formatted.append(f"Assistant: {interaction['answer'][:300]}...")
        
        return "\n".join(formatted[-10:])  # Last 5 interactions (10 messages)
    
    def _create_chain(self):
        """Create the RAG chain with memory"""
        
        # Enhanced prompt template with memory
        template = """You are a highly intelligent AI assistant specialized in answering questions based on provided documents. You have access to conversation history to maintain context.

CONVERSATION HISTORY:
{chat_history}

RETRIEVED CONTEXT FROM DOCUMENTS:
{context}

CURRENT QUESTION:
{question}

INSTRUCTIONS:
1. **Primary Source**: Answer based PRIMARILY on the retrieved context above
2. **Conversation Awareness**: Use conversation history to understand follow-up questions and maintain context
3. **Clarity on Limitations**: If the answer is not in the retrieved context, clearly state: "Based on the provided documents, I don't have enough information to answer this question."
4. **Source Attribution**: When possible, mention which document or section your answer comes from
5. **Conversational**: Maintain a natural, helpful tone. For follow-up questions, acknowledge previous context
6. **Conciseness**: Be clear and concise, but comprehensive enough to fully answer the question
7. **Accuracy**: Never make up information. Only use what's in the context or general knowledge when appropriate

ANSWER:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Create chain with memory integration
        def get_chat_history(_):
            """Get formatted chat history"""
            history = self.memory.get_conversation_history()
            return self._format_chat_history(history[-5:])  # Last 5 interactions
        
        chain = (
            {
                "context": self.retriever | RunnableLambda(self._format_docs),
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(get_chat_history)
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("✅ RAG chain with memory created")
        return chain
    
    def query(self, question: str, max_retries: int = 3) -> Dict:
        """
        Query the RAG system with memory and retry logic
        
        Args:
            question: User question
            max_retries: Maximum number of retries for rate limits
            
        Returns:
            Dict with answer and source documents
        """
        logger.info(f"\n🔍 Processing query: {question}")
        
        # Retrieve relevant documents
        try:
            docs = self.retriever.invoke(question)
            logger.info(f"Retrieved {len(docs)} documents")
        except Exception as e:
            logger.error(f"❌ Error retrieving documents: {str(e)}")
            raise
        
        # Try to get answer with retries
        for attempt in range(max_retries):
            try:
                answer = self.chain.invoke(question)
                logger.info(f"Generated answer: {len(answer)} characters")
                
                # Add to memory
                self.memory.add_interaction(
                    question=question,
                    answer=answer,
                    sources=[{
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "N/A")
                    } for doc in docs]
                )
                
                return {
                    "answer": answer,
                    "source_documents": docs,
                    "memory_stats": self.memory.get_memory_statistics()
                }
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 3  # Exponential backoff: 3s, 6s, 9s
                        logger.warning(f"⚠️ Rate limit hit. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("❌ Rate limit exceeded after all retries")
                        raise Exception(
                            "Rate limit exceeded. Please wait a few minutes and try again. "
                            "You can also:\n"
                            "1. Switch to a different model (gemini-1.5-flash or gemini-1.5-pro)\n"
                            "2. Wait for your quota to reset\n"
                            "3. Use OpenAI instead (requires API key)"
                        )
                else:
                    # Different error, raise immediately
                    logger.error(f"❌ Error generating answer: {str(e)}")
                    raise
    
    def query_with_sources(self, question: str) -> Dict:
        """Query with formatted source information"""
        try:
            response = self.query(question)
            
            # Format sources
            sources = []
            if response["source_documents"]:
                for doc in response["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "N/A"),
                        "file_type": doc.metadata.get("file_type", "Unknown")
                    }
                    sources.append(source_info)
            
            return {
                "answer": response["answer"],
                "sources": sources,
                "memory_stats": response["memory_stats"]
            }
            
        except Exception as e:
            logger.error(f"❌ Error in query_with_sources: {str(e)}")
            raise
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.memory.get_conversation_history()
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear_memory()
        logger.info("Conversation memory cleared")
    
    def export_conversation(self) -> List[Dict]:
        """Export conversation for saving"""
        return self.memory.export_history()
    
    def get_memory_statistics(self) -> Dict:
        """Get memory usage statistics"""
        return self.memory.get_memory_statistics()