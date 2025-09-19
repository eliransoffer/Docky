# memory/conversation_manager.py
from datetime import datetime
from typing import Dict, List, Any, Optional
from .token_counter import TokenCounter

class ConversationManager:
    """Manages conversation history with automatic summarization"""
    
    def __init__(self, llm, max_tokens: int = 500, max_recent_exchanges: int = 3):
        self.llm = llm
        self.max_tokens = max_tokens
        self.max_recent_exchanges = max_recent_exchanges
        self.conversation_history = []
        self.summary = ""
        self.token_counter = TokenCounter()
    
    def add_exchange(self, question: str, answer: str, sources: Optional[List[Dict]] = None):
        """Add a Q&A exchange to history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        exchange = {
            'timestamp': timestamp,
            'question': question,
            'answer': answer,
            'sources': sources or [],
            'tokens': self.token_counter.count_tokens(f"{question} {answer}")
        }
        
        self.conversation_history.append(exchange)
        self._manage_history_size()
    
    def get_context_for_prompt(self) -> str:
        """Get conversation context for the prompt"""
        context = ""
        
        # Add summary if available
        if self.summary:
            context += f"{self.summary}\n\n"
        
        # Add recent conversation history
        if self.conversation_history:
            context += "Recent conversation:\n"
            recent_exchanges = self.conversation_history[-self.max_recent_exchanges:]
            for ex in recent_exchanges:
                context += f"[{ex['timestamp']}] Human: {ex['question']}\n"
                context += f"Assistant: {ex['answer'][:150]}...\n\n"
        
        return context
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        total_exchanges = len(self.conversation_history)
        total_tokens = sum(ex['tokens'] for ex in self.conversation_history)
        has_summary = bool(self.summary)
        
        return {
            'total_exchanges': total_exchanges,
            'total_tokens': total_tokens,
            'has_summary': has_summary,
            'summary_length': len(self.summary) if self.summary else 0
        }
    
    def get_summary_info(self) -> Dict[str, Any]:
        """Get detailed summary and recent history info"""
        return {
            'summary': self.summary,
            'recent_history': self.conversation_history[-self.max_recent_exchanges:],
            'stats': self.get_conversation_stats()
        }
    
    def _manage_history_size(self):
        """Summarize history if it exceeds token limit"""
        total_tokens = sum(ex['tokens'] for ex in self.conversation_history)
        
        if total_tokens > self.max_tokens and len(self.conversation_history) > 1:
            print("📝 Conversation getting long, creating summary...")
            
            # Take first half of conversations to summarize
            split_point = len(self.conversation_history) // 2
            to_summarize = self.conversation_history[:split_point]
            to_keep = self.conversation_history[split_point:]
            
            # Create summary of older conversations
            summary_text = self._create_summary(to_summarize)
            
            # Update summary and keep recent conversations
            if self.summary:
                self.summary = f"{self.summary}\n\n{summary_text}"
            else:
                self.summary = summary_text
                
            self.conversation_history = to_keep
            print("✅ Summary created, recent conversation history maintained")
    
    def _create_summary(self, exchanges: List[Dict]) -> str:
        """Create a summary of conversation exchanges"""
        summary_prompt = f"""Summarize this conversation history in 2-3 sentences, focusing on the main topics discussed and key information provided:

Conversation:
"""
        
        for ex in exchanges:
            summary_prompt += f"Q: {ex['question']}\nA: {ex['answer'][:200]}...\n\n"
        
        try:
            summary_response = self.llm.invoke(summary_prompt)
            return f"Previous conversation summary: {summary_response.content}"
        except Exception as e:
            # Fallback summary
            topics = [ex['question'][:50] for ex in exchanges]
            return f"Previous conversation covered topics: {', '.join(topics)}"