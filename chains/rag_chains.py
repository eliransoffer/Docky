# chains/rag_chain.py
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI

class RAGChainBuilder:
    """Builds and manages RAG chains with memory awareness"""
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
    
    def create_memory_aware_prompt(self) -> ChatPromptTemplate:
        """Create prompt template that handles conversation memory"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on provided context and conversation history.

INSTRUCTIONS:
1. Use the document context to provide accurate, well-cited answers
2. The conversation context contains either:
   - A SUMMARY of previous discussions (this replaces old exchanges)
   - Recent conversation exchanges (only the most recent ones)
3. Reference the summary when relevant (e.g., "As we discussed previously...")
4. Don't ask about missing details that might be in the summary
5. Always cite sources using [Page X] format
6. Build naturally on the provided context

CITATION FORMAT:
- Use [Page X] immediately after claims
- Include multiple pages if using multiple sources: [Pages X, Y, Z]
- Be specific about which information comes from which page

CONTEXT USAGE:
- If there's a summary, it represents our complete conversation history up to recent exchanges
- Don't assume information not in the context, but acknowledge what we've covered before
- Be conversational and natural while maintaining accuracy"""),
            
            ("human", """Conversation Context:
{conversation_context}

Document Context:
{context}

Current Question: {input}

Please provide a comprehensive answer that considers both the document context and our conversation history.""")
        ])
    
    def create_basic_prompt(self) -> ChatPromptTemplate:
        """Create basic prompt template without conversation memory"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on provided document context.

INSTRUCTIONS:
1. Use only the provided document context to answer questions
2. Always cite sources using [Page X] format
3. If the context doesn't contain enough information, say so clearly
4. Be specific and accurate in your responses

CITATION FORMAT:
- Use [Page X] immediately after claims
- Include multiple pages if using multiple sources: [Pages X, Y, Z]
- Be specific about which information comes from which page"""),
            
            ("human", """Document Context:
{context}

Question: {input}

Please provide a comprehensive answer based on the document context.""")
        ])
    
    def create_rag_chain(self, retriever, use_memory: bool = True):
        """Create complete RAG chain"""
        try:
            # Choose appropriate prompt based on memory usage
            if use_memory:
                prompt = self.create_memory_aware_prompt()
            else:
                prompt = self.create_basic_prompt()
            
            # Create document chain
            document_chain = create_stuff_documents_chain(self.llm, prompt)
            
            # Create retrieval chain
            rag_chain = create_retrieval_chain(retriever, document_chain)
            
            return rag_chain
            
        except Exception as e:
            raise ValueError(f"Failed to create RAG chain: {str(e)}")
    
    def create_summarization_prompt(self) -> str:
        """Create prompt for conversation summarization"""
        return """Summarize this conversation history in 2-3 sentences, focusing on the main topics discussed and key information provided:

Conversation:
{conversation_text}

Provide a concise summary that captures the essential points discussed."""