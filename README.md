# RAG System with Conversation Memory

A modular RAG (Retrieval-Augmented Generation) system that maintains conversation history with automatic summarization. The system can answer questions about PDF documents while remembering previous conversations.

## Features

- **PDF Document Processing**: Load and chunk PDF documents for retrieval
- **Conversation Memory**: Maintains conversation history with automatic summarization
- **Intelligent Retrieval**: Uses Chroma vector database for semantic search
- **Context-Aware Responses**: Builds on previous conversation exchanges
- **Modular Architecture**: Clean separation of concerns for easy maintenance
- **Configurable**: Easily adjustable parameters for different use cases


## Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```bash
   
   GOOGLE_API_KEY=your_google_api_key_here
   LANGCHAIN_TRACING_V2="true"
   LANGCHAIN_ENDPOINT="https://eu.smith.langchain.com/api"
   LANGCHAIN_API_KEY="your_langchain_api_key_here"
   LANGCHAIN_PROJECT="Docky_api"
   LANGSMITH_TRACING="true"
   LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
   LANGSMITH_API_KEY="your_langsmith_api_key_here"
   LANGSMITH_PROJECT="docky_api"
   ```

## Usage

### Basic Usage

```bash
python main.py path/to/your/document.pdf
```

### Advanced Usage

```bash
# Custom memory limit
python main.py document.pdf --memory-tokens 1000

# Custom vector store location
python main.py document.pdf --persist-dir ./my_vector_db

# Custom chunking parameters
python main.py document.pdf --chunk-size 1500 --chunk-overlap 300
```

## Configuration

The system can be configured through command-line arguments or by modifying `config/settings.py`:

```python
@dataclass
class RAGConfig:
    # Model settings
    embedding_model: str = "models/gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash"
    
    # Document processing
    chunk_size: int = 2000
    chunk_overlap: int = 400
    
    # Memory settings
    memory_tokens: int = 500
    max_recent_exchanges: int = 3
    
    # Vector store
    retrieval_k: int = 6
```

## How It Works

### 1. Document Processing
- Loads PDF documents using PyPDFLoader
- Splits documents into overlapping chunks for better retrieval
- Creates embeddings using Google's Generative AI
- Stores in Chroma vector database for persistent storage

### 2. Conversation Memory
- Tracks conversation exchanges with token counting
- Automatically summarizes older conversations when memory limit is reached
- Maintains recent exchanges for immediate context
- Provides conversation statistics and summaries

### 3. RAG Chain
- Uses LangChain's retrieval chain architecture
- Context-aware prompts that consider conversation history
- Proper citation format with page numbers
- Handles both memory-aware and basic question answering

### 4. Interactive Interface
- Clean CLI with helpful commands
- Real-time conversation statistics
- Easy access to system information
- Graceful error handling

## Example Conversation

```
❓ Your question: What are the main topics covered in this document?

🧠 Thinking with conversation context...

**Answer:** Based on the document, the main topics covered include... [Page 1, 3]

**💬 Conversation Stats:** 1 exchanges, 45 tokens

❓ Your question: Can you elaborate on the first topic you mentioned?

🧠 Thinking with conversation context...

**Answer:** As we discussed previously, the first topic was... [Page 1]

**💬 Conversation Stats:** 2 exchanges, 89 tokens
```

## Architecture Benefits

### Modularity
- Each component has a single responsibility
- Easy to test individual modules
- Clear interfaces between components
- Simple to extend or modify specific functionality

### Maintainability
- Smaller files are easier to understand
- Clear dependency relationships
- Consistent error handling
- Comprehensive logging and feedback

### Flexibility
- Easy to swap implementations
- Configurable through settings
- Multiple interface options (CLI, could add web/API)
- Extensible for new document types or models

## Troubleshooting

### Common Issues

1. **Missing API Key**
   ```
   Error: GOOGLE_API_KEY environment variable is required
   ```
   **Solution**: Set your Google API key in the `.env` file

2. **PDF Loading Error**
   ```
   Error: Failed to load PDF 'document.pdf': [Errno 2] No such file or directory
   ```
   **Solution**: Check the PDF path and ensure the file exists

3. **Memory Warnings**
   ```
   📝 Conversation getting long, creating summary...
   ```
   **Info**: This is normal behavior when conversation exceeds token limit

4. **Vector Store Issues**
   ```
   Error: Failed to initialize vector store
   ```
   **Solution**: Check write permissions for the persist directory

## Contributing

To extend the system:

1. **Add new document types**: Extend `DocumentLoader` class
2. **Add new interfaces**: Create new modules in `cli/` or add `web/` directory
3. **Modify memory behavior**: Customize `ConversationManager` class
4. **Change prompts**: Modify templates in `RAGChainBuilder`

## Dependencies

- **LangChain**: Framework for LLM applications
- **ChromaDB**: Vector database for document embeddings
- **Google Generative AI**: Embeddings and language model
- **PyPDF**: PDF document processing
- **tiktoken**: Token counting for memory management

## License

This project is provided as-is for educational and development purposes.