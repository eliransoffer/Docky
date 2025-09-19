# memory/token_counter.py
import tiktoken
from typing import Optional

class TokenCounter:
    """Handles token counting for text"""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize token counter with specified encoding"""
        self.encoding_name = encoding_name
        self.tokenizer: Optional[tiktoken.Encoding] = None
        
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            print(f"Warning: Could not load tokenizer {encoding_name}: {e}")
            print("Using fallback token estimation")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation: ~4 characters per token
            return len(text) // 4
    
    def estimate_tokens_for_exchanges(self, exchanges: list) -> int:
        """Estimate total tokens for a list of exchanges"""
        total_text = ""
        for ex in exchanges:
            total_text += f"{ex.get('question', '')} {ex.get('answer', '')}"
        
        return self.count_tokens(total_text)
    
    def truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        if self.count_tokens(text) <= max_tokens:
            return text
        
        if self.tokenizer:
            # Encode and truncate tokens
            tokens = self.tokenizer.encode(text)
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens)
        else:
            # Fallback: character-based truncation
            max_chars = max_tokens * 4  # Rough estimation
            return text[:max_chars] if len(text) > max_chars else text