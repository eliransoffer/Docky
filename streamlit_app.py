import streamlit as st
import os
import tempfile
from pathlib import Path
import time
import asyncio
import threading
import hashlib
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from config.settings import RAGConfig
from core.rag_system import RAGWithMemory

# Page config
st.set_page_config(
    page_title="Docky",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat appearance
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-info {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
    }
    .stats-box {
        background-color: #e8f5e8;
        padding: 0.8rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .document-info {
        background-color: #fff3e0;
        padding: 0.8rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def safe_async_call(async_func, *args, **kwargs):
    """
    Safely call an async function from a synchronous context.
    This handles the event loop creation for Streamlit's threading environment.
    """
    def run_in_thread():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()
    
    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        # If we're in a loop, we need to run in a separate thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()
    except RuntimeError:
        # No event loop running, safe to create one
        return run_in_thread()

def get_file_hash(uploaded_file):
    """Generate a hash of the uploaded file to detect changes"""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()

def initialize_session_state():
    """Initialize session state variables"""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "current_document_hash" not in st.session_state:
        st.session_state.current_document_hash = None
    if "current_document_name" not in st.session_state:
        st.session_state.current_document_name = None

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None
    
def ensure_event_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def clear_rag_system():
    """Completely clear the RAG system and reset state"""
    if st.session_state.rag_system:
        try:
            # Clear conversation history
            st.session_state.rag_system.clear_conversation_history()
            
            # If there's a method to clear vector store, call it
            if hasattr(st.session_state.rag_system, 'clear_vector_store'):
                st.session_state.rag_system.clear_vector_store()
            
            # If there's a cleanup method, call it
            if hasattr(st.session_state.rag_system, 'cleanup'):
                st.session_state.rag_system.cleanup()
                
        except Exception as e:
            st.warning(f"Error during cleanup: {str(e)}")
    
    # Reset all session state
    st.session_state.rag_system = None
    st.session_state.messages = []
    st.session_state.document_processed = False
    st.session_state.current_document_hash = None
    st.session_state.current_document_name = None
    
def setup_rag_system(pdf_path, config):
    """Initialize and setup RAG system"""
    try:
        ensure_event_loop()  # make sure this thread has a loop
        with st.spinner("Initializing document..."):
            rag_system = RAGWithMemory(pdf_path, config)
            
            # Load and process documents
            with st.spinner("Processing document..."):
                result = rag_system.load_and_process_documents()
                st.success(f"✅ {result}")
            
            # Setup chain
            with st.spinner("Setting up RAG chain..."):
                rag_system.setup_chain()
                st.success("✅ RAG system ready!")
            
            return rag_system
            
    except Exception as e:
        st.error(f"❌ Error setting up RAG system: {str(e)}")
        return None

def display_message(message_type, content, sources=None, timestamp=None):
    """Display a chat message with proper formatting"""
    if message_type == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Display sources if available
        if sources:
            with st.expander("📚 View Sources", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    <div class="source-info">
                        <strong>Source {i}:</strong> Page {source['page']} (Chunk {source['chunk_id']})<br>
                        <em>Preview:</em> "{source['content_preview']}"
                    </div>
                    """, unsafe_allow_html=True)

def display_conversation_stats(stats):
    """Display conversation statistics"""
    st.markdown(f"""
    <div class="stats-box">
        <strong>💬 Conversation Stats:</strong><br>
        📊 {stats['total_exchanges']} exchanges | 🔤 {stats['total_tokens']} tokens
        {f" | 📝 Summary available ({stats['summary_length']} chars)" if stats['has_summary'] else ""}
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.title("Docky chat")
    st.markdown("*Upload a PDF and chat with your document while I remember our conversation!*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        st.subheader("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to chat with"
        )
        
        # Show current document info
        if st.session_state.current_document_name:
            st.markdown(f"""
            <div class="document-info">
                <strong>📋 Current Document:</strong><br>
                {st.session_state.current_document_name}
            </div>
            """, unsafe_allow_html=True)
        
        # Configuration settings
        st.subheader("🔧 Settings")
        memory_tokens = st.slider("Memory Tokens", 100, 2000, 500, 100)
        chunk_size = st.slider("Chunk Size", 500, 3000, 2000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 800, 400, 50)
        retrieval_k = st.slider("Retrieval Chunks", 3, 10, 6)
        
        # System info
        if st.session_state.rag_system:
            st.subheader("📊 System Info")
            try:
                info = st.session_state.rag_system.get_system_info()
                
                st.metric("Status", "🟢 Ready" if info['status'] == 'ready' else "🟡 Setup")
                
                if 'vector_store' in info and info['vector_store'].get('initialized'):
                    vs_info = info['vector_store']
                    st.metric("Document Chunks", vs_info.get('document_count', 0))
                    if 'unique_pages' in vs_info:
                        st.metric("Unique Pages", vs_info['unique_pages'])
            except Exception as e:
                st.warning(f"Could not load system info: {str(e)}")
        
        # Action buttons
        st.subheader("Actions")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🧹 Clear Chat", use_container_width=True):
                if st.session_state.rag_system:
                    try:
                        st.session_state.rag_system.clear_conversation_history()
                        st.session_state.messages = []
                        st.success("Chat history cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing history: {str(e)}")
        
        with col_b:
            if st.button("🔄 Reset All", use_container_width=True):
                try:
                    clear_rag_system()
                    st.success("✅ Complete system reset!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Reset failed: {str(e)}")
                    st.info("💡 Try refreshing the page if issues persist")
        
        if st.button("Show Conversation Summary", use_container_width=True):
            if st.session_state.rag_system:
                try:
                    summary_info = st.session_state.rag_system.get_conversation_summary()
                    
                    with st.expander("📝 Conversation Summary", expanded=True):
                        if summary_info['summary']:
                            st.write("**Summary:**")
                            st.write(summary_info['summary'])
                        else:
                            st.info("No summary available yet.")
                        
                        if summary_info['recent_history']:
                            st.write("**Recent History:**")
                            for ex in summary_info['recent_history'][-3:]:
                                st.write(f"**[{ex['timestamp']}]** {ex['question'][:60]}...")
                except Exception as e:
                    st.error(f"Error getting summary: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Check if we need to process a new document
        needs_processing = False
        
        if uploaded_file:
            file_hash = get_file_hash(uploaded_file)
            
            # Check if this is a new document
            if (file_hash != st.session_state.current_document_hash or 
                not st.session_state.document_processed):
                needs_processing = True
                
                # Clear previous system if switching documents
                if (st.session_state.current_document_hash and 
                    file_hash != st.session_state.current_document_hash):
                    st.info("🔄 New document detected. Clearing previous document...")
                    clear_rag_system()
        
        # Document processing
        if uploaded_file and needs_processing:
            # Save uploaded file
            pdf_path = save_uploaded_file(uploaded_file)
            
            if pdf_path:
                try:
                    # Create config
                    config = RAGConfig(
                        memory_tokens=memory_tokens,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        retrieval_k=retrieval_k
                    )
                    
                    # Setup RAG system
                    rag_system = setup_rag_system(pdf_path, config)
                    
                    if rag_system:
                        st.session_state.rag_system = rag_system
                        st.session_state.document_processed = True
                        st.session_state.current_document_hash = get_file_hash(uploaded_file)
                        st.session_state.current_document_name = uploaded_file.name
                        st.session_state.messages = []
                        
                        # Clean up temp file
                        try:
                            os.unlink(pdf_path)
                        except:
                            pass
                        
                        st.rerun()
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    # Clean up temp file on error
                    try:
                        os.unlink(pdf_path)
                    except:
                        pass
        
        # Chat interface
        if st.session_state.rag_system and st.session_state.document_processed:
            st.subheader("💬 Chat with your document")
            
            # Display conversation history
            for message in st.session_state.messages:
                display_message(
                    message["type"],
                    message["content"],
                    message.get("sources"),
                    message.get("timestamp")
                )
            
            # Chat input
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Your question:",
                    placeholder="Ask a question about your document...",
                    height=100,
                    key="user_question"
                )
                
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    submit_button = st.form_submit_button("Send", use_container_width=True)
            
            # Process user input
            if submit_button and user_input.strip():
                # Add user message to history
                st.session_state.messages.append({
                    "type": "user",
                    "content": user_input,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                
                # Get response from RAG system
                try:
                    with st.spinner("Thinking with conversation context..."):
                        # Check if ask_with_memory is async or sync
                        ask_method = st.session_state.rag_system.ask_with_memory
                        
                        if asyncio.iscoroutinefunction(ask_method):
                            # Use safe async wrapper for async methods
                            result = safe_async_call(ask_method, user_input)
                        else:
                            # Call synchronously for sync methods
                            result = ask_method(user_input)
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "type": "assistant",
                        "content": result['answer'],
                        "sources": result.get('sources', []),
                        "timestamp": time.strftime("%H:%M:%S")
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    # Log the full error for debugging
                    import traceback
                    st.error(f"Full error: {traceback.format_exc()}")
        
        else:
            # Welcome message
            st.info("👆 Please upload a PDF document in the sidebar to start chatting!")
            
            # Example usage
            with st.expander("ℹ️ How to use this app", expanded=True):
                st.markdown("""
                **Steps:**
                1. **Upload a PDF** in the sidebar
                2. **Wait** for the document to be processed
                3. **Ask questions** about your document
                4. **Continue the conversation** - I'll remember what we discussed!
                
                **Features:**
                - **Memory**: I remember our conversation and build on previous exchanges
                - **Sources**: See exactly which parts of the document I'm referencing
                - **Configurable**: Adjust memory, chunk size, and other parameters
                - **Stats**: Track conversation history and token usage
                
                **New Document Handling:**
                - Upload a new PDF to automatically switch documents
                - Previous conversations are cleared when switching
                - Use "Reset All" button to completely clear the system
                - Use "Clear Chat" to keep document but clear conversation
                
                **Troubleshooting:**
                - If you get async errors, the app will handle them automatically
                - Check the error messages for specific issues
                - Try the "Reset All" button if things get stuck
                """)
    
    with col2:
        # Real-time stats
        if st.session_state.rag_system:
            st.subheader("📈 Live Stats")
            
            try:
                conv_stats = st.session_state.rag_system.conversation_manager.get_conversation_stats()
                
                st.metric("Total Exchanges", conv_stats['total_exchanges'])
                st.metric("Total Tokens", conv_stats['total_tokens'])
                
                # Progress bar for memory usage
                memory_usage = conv_stats['total_tokens'] / memory_tokens
                st.progress(min(memory_usage, 1.0))
                st.caption(f"Memory Usage: {memory_usage:.1%}")
                
                if conv_stats['has_summary']:
                    st.success("📝 Summary created")
                    st.caption(f"Summary: {conv_stats['summary_length']} chars")
                    
            except Exception as e:
                st.warning(f"Could not load stats: {str(e)}")
        
        # Tips
        st.subheader("💡 Tips")
        st.markdown("""
        - **Upload new PDFs** anytime to switch documents
        - **Follow up questions** work great with memory
        - **Ask for clarification** on previous answers
        - **Reference earlier topics** naturally
        - **Check sources** for accuracy
        - **Use Reset All** for a fresh start
        - **Adjust settings** in sidebar for better results
        """)

if __name__ == "__main__":
    main()