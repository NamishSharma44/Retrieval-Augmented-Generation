"""
Memory Manager for Conversational RAG
Handles conversation history and context management
"""
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages conversation history and context"""
    
    def __init__(self, k: int = 5, memory_type: str = "window"):
        """
        Initialize memory manager
        
        Args:
            k: Number of previous message pairs to remember
            memory_type: 'window' for fixed window or 'summary' for summarization
        """
        self.k = k
        self.memory_type = memory_type
        self.conversation_history: List[Dict] = []
        
        logger.info(f"MemoryManager initialized: type={memory_type}, k={k}")
    
    def add_interaction(self, question: str, answer: str, sources: Optional[List] = None):
        """Add a question-answer interaction to memory"""
        # Add to conversation history
        interaction = {
            "question": question,
            "answer": answer,
            "sources": sources or []
        }
        self.conversation_history.append(interaction)
        
        logger.debug(f"Added interaction to memory. Total interactions: {len(self.conversation_history)}")
    
    def get_conversation_history(self) -> List[Dict]:
        """Get full conversation history"""
        return self.conversation_history
    
    def get_recent_context(self, k: Optional[int] = None) -> str:
        """Get recent conversation context as formatted string"""
        k = k or self.k
        recent = self.conversation_history[-k:]
        
        if not recent:
            return ""
        
        context_parts = []
        for i, interaction in enumerate(recent, 1):
            context_parts.append(f"Previous Q{i}: {interaction['question']}")
            context_parts.append(f"Previous A{i}: {interaction['answer'][:200]}...")
        
        return "\n".join(context_parts)
    
    def get_langchain_memory(self):
        """Get conversation history for LangChain integration"""
        return self.conversation_history[-self.k:]
    
    def clear_memory(self):
        """Clear all conversation history"""
        self.conversation_history = []
        logger.info("Memory cleared")
    
    def get_memory_statistics(self) -> Dict:
        """Get memory usage statistics"""
        total_interactions = len(self.conversation_history)
        total_chars = sum(
            len(i['question']) + len(i['answer']) 
            for i in self.conversation_history
        )
        
        return {
            "total_interactions": total_interactions,
            "total_characters": total_chars,
            "memory_window": self.k,
            "recent_interactions": min(self.k, total_interactions)
        }
    
    def format_history_for_context(self) -> List[Dict]:
        """Format history for LLM context"""
        formatted = []
        for interaction in self.conversation_history[-self.k:]:
            formatted.append({
                "role": "user",
                "content": interaction["question"]
            })
            formatted.append({
                "role": "assistant",
                "content": interaction["answer"]
            })
        return formatted
    
    def export_history(self) -> List[Dict]:
        """Export conversation history for saving/analysis"""
        return self.conversation_history.copy()
    
    def import_history(self, history: List[Dict]):
        """Import conversation history"""
        self.conversation_history = history
        logger.info(f"Imported {len(history)} interactions")