"""
Production-Grade RAG System - Streamlit Interface
Enhanced with memory, better UX, and robust error handling
"""
import streamlit as st
import os
import tempfile
import logging
import shutil
from pathlib import Path
import json
from datetime import datetime

# Import configuration first
from config import Config

# Import modules
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from core.rag_chain import RAGChain

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Document QA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .source-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'vectorstore': None,
        'rag_chain': None,
        'chat_history': [],
        'documents_processed': False,
        'processed_files': [],
        'api_configured': False,
        'system_ready': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Check API configuration
def check_api_configuration():
    """Check if API keys are configured"""
    api_status = Config.validate_api_keys()
    
    if not any(api_status.values()):
        st.error("⚠️ No API keys found! Please configure your .env file.")
        st.code("""
# Create a .env file in your project directory with:
GOOGLE_API_KEY=your_google_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here

# Optional configurations:
LLM_PROVIDER=google  # or 'openai'
GOOGLE_MODEL=gemini-2.0-flash-exp
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
        """)
        return False
    
    st.session_state.api_configured = True
    return True

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("⚙️ Configuration")
    
    # API Status
    api_status = Config.validate_api_keys()
    
    with st.expander("🔑 API Status", expanded=not st.session_state.api_configured):
        if api_status['google']:
            st.success("✅ Google API Key: Configured")
        else:
            st.warning("⚠️ Google API Key: Not found")
        
        if api_status['openai']:
            st.success("✅ OpenAI API Key: Configured")
        else:
            st.info("ℹ️ OpenAI API Key: Not configured")
        
        if not any(api_status.values()):
            st.error("❌ No API keys configured!")
            st.info("💡 Add your API key to the .env file")
    
    st.markdown("---")
    
    # Model Configuration
    st.subheader("🤖 Model Settings")
    
    available_providers = []
    if api_status['google']:
        available_providers.append("google")
    if api_status['openai']:
        available_providers.append("openai")
    
    if available_providers:
        provider = st.selectbox(
            "LLM Provider",
            available_providers,
            index=0 if Config.DEFAULT_LLM_PROVIDER in available_providers else 0
        )
        
        if provider == "google":
            model_options = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
            default_model = Config.GOOGLE_MODEL if Config.GOOGLE_MODEL in model_options else model_options[0]
        else:
            model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
            default_model = Config.OPENAI_MODEL if Config.OPENAI_MODEL in model_options else model_options[0]
        
        model_name = st.selectbox(
            "Model",
            model_options,
            index=model_options.index(default_model) if default_model in model_options else 0
        )
    else:
        st.error("No LLM providers available!")
        provider = None
        model_name = None
    
    st.markdown("---")
    
    # Document Processing Settings
    st.subheader("📄 Document Settings")
    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=Config.DEFAULT_CHUNK_SIZE,
        step=100,
        help="Size of text chunks for processing"
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=Config.DEFAULT_CHUNK_OVERLAP,
        step=50,
        help="Overlap between consecutive chunks"
    )
    
    st.markdown("---")
    
    # Retrieval Settings
    st.subheader("🔍 Retrieval Settings")
    num_results = st.slider(
        "Results to Retrieve",
        min_value=1,
        max_value=10,
        value=Config.DEFAULT_K,
        help="Number of relevant chunks to retrieve"
    )
    
    memory_k = st.slider(
        "Conversation Memory",
        min_value=1,
        max_value=10,
        value=Config.MEMORY_K,
        help="Number of previous messages to remember"
    )
    
    # Detect provider/model changes and reinitialize RAG chain
    if st.session_state.documents_processed and st.session_state.rag_chain:
        current_provider = st.session_state.get('current_provider')
        current_model = st.session_state.get('current_model')
        
        if current_provider != provider or current_model != model_name:
            st.info("⚙️ Model configuration changed. Click 'Reinitialize' to apply.")
            if st.button("🔄 Reinitialize RAG Chain", type="primary"):
                with st.spinner("Reinitializing with new model..."):
                    try:
                        api_key = Config.GOOGLE_API_KEY if provider == "google" else Config.OPENAI_API_KEY
                        
                        st.session_state.rag_chain = RAGChain(
                            vectorstore=st.session_state.vectorstore,
                            provider=provider,
                            model_name=model_name,
                            api_key=api_key,
                            k=num_results,
                            memory_k=memory_k
                        )
                        
                        st.session_state.current_provider = provider
                        st.session_state.current_model = model_name
                        
                        st.success(f"✅ Switched to {provider}/{model_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # System Status
    with st.expander("📊 System Status", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            status_icon = "✅" if st.session_state.documents_processed else "⏸️"
            st.metric("System Status", status_icon)
        
        with col2:
            doc_count = len(st.session_state.processed_files)
            st.metric("Documents", doc_count)
        
        if st.session_state.vectorstore:
            try:
                # Try to get count from the vectorstore
                if hasattr(st.session_state.vectorstore, '_collection'):
                    count = st.session_state.vectorstore._collection.count()
                    st.metric("Vector Chunks", count)
                elif hasattr(st.session_state.vectorstore, 'vectorstore'):
                    count = st.session_state.vectorstore.vectorstore._collection.count()
                    st.metric("Vector Chunks", count)
            except:
                st.metric("Vector Chunks", "Active")
        
        if st.session_state.rag_chain:
            mem_stats = st.session_state.rag_chain.get_memory_statistics()
            st.metric("Chat History", mem_stats.get('total_interactions', 0))
    
    st.markdown("---")
    
    # Actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Data", use_container_width=True, type="secondary"):
            try:
                # Clear vector store
                if st.session_state.vectorstore:
                    try:
                        # Delete the ChromaDB directory
                        import shutil
                        if os.path.exists(Config.VECTOR_STORE_DIR):
                            shutil.rmtree(Config.VECTOR_STORE_DIR)
                        logger.info("✅ Vector store deleted")
                    except Exception as e:
                        logger.warning(f"Could not delete vector store: {e}")
                
                # Reset session state
                st.session_state.vectorstore = None
                st.session_state.rag_chain = None
                st.session_state.chat_history = []
                st.session_state.documents_processed = False
                st.session_state.processed_files = []
                
                st.success("✅ Data cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("🔄 Clear Chat", use_container_width=True, type="secondary"):
            if st.session_state.rag_chain:
                st.session_state.rag_chain.clear_memory()
            st.session_state.chat_history = []
            st.success("✅ Chat cleared!")
            st.rerun()

# Main Content
st.markdown('<h1 class="main-header">📚 RAG Document QA System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload documents and ask questions with AI-powered context awareness</p>', unsafe_allow_html=True)

# Check API configuration
if not check_api_configuration():
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "💬 Chat", "📊 Analytics", "ℹ️ About"])

# Tab 1: Document Upload
with tab1:
    st.header("📤 Upload Your Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'doc', 'txt'],
            accept_multiple_files=True,
            help=f"Supported: PDF, Word, Text (Max {Config.MAX_FILE_SIZE_MB}MB each)"
        )
    
    with col2:
        if st.session_state.documents_processed:
            st.success("✅ Documents Ready")
            st.info(f"📁 {len(st.session_state.processed_files)} files loaded")
    
    if uploaded_files:
        st.markdown("### 📋 Selected Files")
        for file in uploaded_files:
            size_mb = file.size / (1024 * 1024)
            st.text(f"📄 {file.name} ({size_mb:.2f} MB)")
        
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            if not provider:
                st.error("❌ No LLM provider configured!")
                st.stop()
            
            with st.spinner("Processing documents... This may take a few minutes."):
                try:
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Save files temporarily
                    status_text.text("📁 Saving uploaded files...")
                    progress_bar.progress(10)
                    
                    temp_files = []
                    for uploaded_file in uploaded_files:
                        suffix = Path(uploaded_file.name).suffix
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            temp_files.append(tmp.name)
                    
                    # Process documents
                    status_text.text("📄 Loading and chunking documents...")
                    progress_bar.progress(25)
                    
                    processor = DocumentProcessor(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    chunks = processor.process_files(temp_files)
                    
                    if not chunks:
                        st.error("❌ No content extracted from documents!")
                        for temp_file in temp_files:
                            os.unlink(temp_file)
                        st.stop()
                    
                    # Create vector store
                    status_text.text("🔢 Creating embeddings...")
                    progress_bar.progress(50)
                    
                    vector_manager = VectorStoreManager(
                        persist_directory=Config.VECTOR_STORE_DIR,
                        collection_name=Config.COLLECTION_NAME,
                        embedding_model=Config.EMBEDDING_MODEL
                    )
                    st.session_state.vectorstore = vector_manager.create_vectorstore(chunks)
                    
                    # Initialize RAG chain
                    status_text.text("🤖 Initializing AI system...")
                    progress_bar.progress(75)
                    
                    api_key = Config.GOOGLE_API_KEY if provider == "google" else Config.OPENAI_API_KEY
                    
                    st.session_state.rag_chain = RAGChain(
                        vectorstore=st.session_state.vectorstore,
                        provider=provider,
                        model_name=model_name,
                        api_key=api_key,
                        k=num_results,
                        memory_k=memory_k
                    )
                    
                    # Store current configuration
                    st.session_state.current_provider = provider
                    st.session_state.current_model = model_name
                    
                    # Complete
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    # Update state
                    st.session_state.documents_processed = True
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    
                    # Cleanup
                    for temp_file in temp_files:
                        os.unlink(temp_file)
                    
                    st.success(f"✅ Successfully processed {len(uploaded_files)} documents with {len(chunks)} chunks!")
                    st.balloons()
                    st.info("👉 Go to the 'Chat' tab to start asking questions!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.exception("Error processing documents")

# Tab 2: Chat Interface
with tab2:
    st.header("💬 Chat with Your Documents")
    
    if not st.session_state.documents_processed:
        st.info("👈 Please upload and process documents first in the 'Upload' tab.")
    else:
        # Display memory statistics
        if st.session_state.rag_chain:
            mem_stats = st.session_state.rag_chain.get_memory_statistics()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💬 Conversations", mem_stats['total_interactions'])
            with col2:
                st.metric("🧠 Memory Window", mem_stats['memory_window'])
            with col3:
                st.metric("📝 Active Memory", mem_stats['recent_interactions'])
            with col4:
                st.metric("🔤 Total Characters", f"{mem_stats['total_characters']:,}")
        
        st.markdown("---")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if "sources" in message and message["sources"]:
                    with st.expander("📚 View Sources", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Source {i}:</strong> {source['source']} 
                                <em>(Page: {source['page']})</em><br>
                                <small>{source['content'][:250]}...</small>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Chat input
        if question := st.chat_input("Ask a question about your documents..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(question)
            
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        response = st.session_state.rag_chain.query_with_sources(question)
                        
                        # Display answer
                        st.markdown(response["answer"])
                        
                        # Display sources
                        if response["sources"]:
                            with st.expander("📚 View Sources", expanded=False):
                                for i, source in enumerate(response["sources"], 1):
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <strong>Source {i}:</strong> {source['source']} 
                                        <em>(Page: {source['page']})</em><br>
                                        <small>{source['content'][:250]}...</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Save to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response["answer"],
                            "sources": response["sources"]
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        logger.exception("Error generating response")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })

# Tab 3: Analytics
with tab3:
    st.header("📊 System Analytics")
    
    if not st.session_state.documents_processed:
        st.info("No data available. Please process documents first.")
    else:
        # Document Statistics
        st.subheader("📄 Document Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", len(st.session_state.processed_files))
        
        with col2:
            if st.session_state.vectorstore:
                try:
                    if hasattr(st.session_state.vectorstore, '_collection'):
                        count = st.session_state.vectorstore._collection.count()
                        st.metric("Vector Chunks", count)
                    elif hasattr(st.session_state.vectorstore, 'vectorstore'):
                        count = st.session_state.vectorstore.vectorstore._collection.count()
                        st.metric("Vector Chunks", count)
                    else:
                        st.metric("Vector Chunks", "Active")
                except:
                    st.metric("Vector Chunks", "Active")
        
        with col3:
            if st.session_state.rag_chain:
                mem_stats = st.session_state.rag_chain.get_memory_statistics()
                st.metric("Chat Interactions", mem_stats['total_interactions'])
        
        # Processed Files
        st.subheader("📁 Processed Files")
        for idx, filename in enumerate(st.session_state.processed_files, 1):
            st.text(f"{idx}. {filename}")
        
        # Conversation History Export
        if st.session_state.rag_chain:
            st.subheader("💾 Export Conversation")
            
            if st.button("📥 Download Chat History"):
                history = st.session_state.rag_chain.export_conversation()
                
                export_data = {
                    "exported_at": datetime.now().isoformat(),
                    "total_interactions": len(history),
                    "conversations": history
                }
                
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Tab 4: About
with tab4:
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ## 🎯 What is RAG?
    
    **Retrieval-Augmented Generation (RAG)** combines:
    - **Retrieval**: Finding relevant information from your documents using semantic search
    - **Generation**: Using Large Language Models to generate accurate, contextual answers
    - **Memory**: Maintaining conversation context for natural, multi-turn interactions
    
    ## 🛠️ Technology Stack
    
    - **LangChain**: Framework for building LLM applications
    - **Google Gemini / OpenAI**: State-of-the-art language models
    - **ChromaDB**: Vector database for semantic search
    - **HuggingFace**: Sentence transformers for embeddings
    - **Streamlit**: Modern web interface
    
    ## ✨ Key Features
    
    - ✅ **Multi-format Support**: PDF, Word, and Text documents
    - ✅ **Conversational Memory**: Maintains context across multiple questions
    - ✅ **Source Attribution**: Cites sources for all answers
    - ✅ **Smart Chunking**: Intelligent document segmentation
    - ✅ **Semantic Search**: Finds the most relevant information
    - ✅ **Production-Ready**: Robust error handling and logging
    - ✅ **Persistent Storage**: Saves embeddings for future use
    - ✅ **Multi-Provider**: Supports Google Gemini and OpenAI
    
    ## 🚀 How to Use
    
    1. **Upload Documents**: Go to Upload tab and select files
    2. **Process**: Click "Process Documents" to create embeddings
    3. **Chat**: Ask questions in the Chat tab with full context awareness!
    
    ## 📝 Tips for Best Results
    
    - Ask specific, focused questions
    - Reference previous questions for context
    - Check sources to verify information
    - Adjust retrieval settings for different document types
    - Use conversation memory for complex, multi-step queries
    
    ## 🔐 Privacy & Security
    
    - All processing happens locally on your machine
    - Documents are processed in memory and temporary files
    - API keys are stored securely in `.env` file
    - No data is sent to third parties except LLM APIs
    """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ using **LangChain**, **Gemini/OpenAI**, and **Streamlit**")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"🔧 Version: 2.0.0")
with col2:
    st.caption(f"🤖 Provider: {provider if provider else 'Not configured'}")
with col3:
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d')}")