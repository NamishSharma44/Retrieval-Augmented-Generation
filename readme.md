# 🚀 RAG Document QA System

A state-of-the-art Retrieval-Augmented Generation (RAG) system with conversational memory, multi-format document support, and enterprise-level features.

**Check the site**: https://retrieval-augmented-generation-7yfyadagf2t53xbkjrhm3q.streamlit.app/

## ✨ Features

- **🧠 Conversational Memory**: Maintains context across multiple questions
- **📄 Multi-Format Support**: PDF, Word (.docx, .doc), and Text files
- **🔍 Semantic Search**: Intelligent retrieval using vector embeddings
- **🤖 Multi-Provider**: Supports Google Gemini and OpenAI models
- **💾 Persistent Storage**: Saves embeddings for fast reloading
- **📊 Analytics Dashboard**: Track usage and conversation history
- **🎨 Modern UI**: Beautiful, responsive Streamlit interface
- **🔐 Secure**: API keys stored in .env file
- **📝 Source Attribution**: Cites sources for all answers
- **⚡ Production-Ready**: Robust error handling and logging

## 🏗️ Architecture

```
rag-system/
├── config.py              # Configuration management
├── app.py                 # Streamlit web interface
├── utils/
│   ├── document_processor.py  # Document loading & chunking
│   └── vector_store.py        # Vector store management
├── core/
│   ├── rag_chain.py          # RAG chain with memory
│   └── memory_manager.py     # Conversation memory
├── .env                   # Environment variables (create this)
├── .env.template         # Template for .env
└── requirements.txt      # Python dependencies
```

## 📋 Prerequisites

- Python 3.9+
- At least one API key:
  - **Google Gemini** (Free tier available): [Get API Key](https://makersuite.google.com/app/apikey)
  - **OpenAI** (Paid): [Get API Key](https://platform.openai.com/api-keys)

## 🚀 Quick Start

### 1. Clone or Download the Project

```bash
git clone <your-repo-url>
cd rag-system
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

```bash
# Copy the template
cp .env.template .env

# Edit .env and add your API key
# For Google Gemini:
GOOGLE_API_KEY=your_google_api_key_here

# OR for OpenAI:
OPENAI_API_KEY=your_openai_api_key_here
```

**Example .env file:**
```env
GOOGLE_API_KEY=...your_key_here
LLM_PROVIDER=google
GOOGLE_MODEL=gemini-2.0-flash-exp
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MEMORY_K=5
```

### 5. Run the Application

```bash
streamlit run app.py
```



## 📖 How to Use

### Step 1: Upload Documents
1. Go to the **📤 Upload** tab
2. Click "Choose files" and select your documents (PDF, Word, or Text)
3. Click **🚀 Process Documents**
4. Wait for processing to complete (20-30 Seconds)

### Step 2: Ask Questions
1. Go to the **💬 Chat** tab
2. Type your question in the chat input
3. Get AI-powered answers with source citations
4. Ask follow-up questions - the system remembers context!

### Step 3: View Analytics
1. Go to the **📊 Analytics** tab
2. See document statistics and conversation history
3. Export chat history as JSON

## 🎯 Key Components

### Document Processor
- **Multiple Loaders**: PDF, DOCX, DOC, TXT with fallback mechanisms
- **Smart Chunking**: Recursive character splitting with overlap
- **Metadata Enrichment**: Source tracking and page numbers
- **Error Handling**: Graceful handling of corrupted or empty files

### Vector Store
- **ChromaDB**: Fast, persistent vector storage
- **HuggingFace Embeddings**: Free, local embeddings (no API needed)
- **Batch Processing**: Efficient handling of large documents
- **Persistence**: Saves embeddings to disk for quick reloading

### RAG Chain
- **Conversational Memory**: Remembers previous interactions
- **Context-Aware**: Uses history to understand follow-up questions
- **Multi-Provider**: Works with Google Gemini or OpenAI
- **Source Attribution**: Always cites sources

### Memory Manager
- **Window Memory**: Configurable conversation history window
- **Export/Import**: Save and load conversation history
- **Statistics**: Track memory usage and conversation metrics

## ⚙️ Configuration Options

### In Sidebar UI:
- **Model Selection**: Choose between different LLM models
- **Chunk Size**: Adjust document chunking (500-2000 characters)
- **Chunk Overlap**: Set overlap between chunks (0-500 characters)
- **Retrieval K**: Number of relevant chunks to retrieve (1-10)
- **Memory K**: Number of previous conversations to remember (1-10)

### In .env File:
- See `.env.template` for all available configuration options

## 🔧 Advanced Usage

### Using Different Models

**Google Gemini:**
```env
LLM_PROVIDER=google
GOOGLE_MODEL=gemini-2.0-flash-exp  # Fast and efficient
# OR
GOOGLE_MODEL=gemini-1.5-pro  # More capable
```

**OpenAI:**
```env
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini  # Cost-effective
# OR
OPENAI_MODEL=gpt-4o  # Most capable
```

### Custom Embedding Models

```env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2  # Better quality
# OR
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Faster
```

### Using GPU for Embeddings

```env
EMBEDDING_DEVICE=cuda  # Requires CUDA-compatible GPU
```

## 📊 Example Queries

**Simple Questions:**
- "What is the main topic of these documents?"
- "Summarize the key findings"
- "What are the conclusions?"

**Follow-up Questions (with memory):**
- User: "What are the benefits of X?"
- Assistant: [Provides answer]
- User: "Can you give me more details about the second benefit?"
- Assistant: [Understands "second benefit" from previous answer]

**Specific Queries:**
- "Find all mentions of [specific term]"
- "Compare X and Y according to the documents"
- "What does page 5 say about [topic]?"

## 🐛 Troubleshooting

### Issue: "No API keys configured"
**Solution:** Make sure you've created `.env` file and added your API key

### Issue: "No content extracted from documents"
**Solution:** 
- Check if PDF is text-based (not scanned image)
- Try a different file format
- Ensure file is not corrupted

### Issue: "Error loading Word document"
**Solution:** System tries multiple loaders. If all fail:
- Save as .txt or PDF format
- Check if file is password-protected

### Issue: "Memory/Performance issues"
**Solution:**
- Reduce chunk size
- Process fewer documents at once
- Increase available RAM

### Issue: "Embeddings taking too long"
**Solution:**
- First run downloads model (~90MB), subsequent runs are faster
- Consider using GPU if available
- Try a smaller embedding model

## 📝 Best Practices

1. **Document Quality**: Use text-based PDFs, not scanned images
2. **File Size**: Keep individual files under 50MB for optimal performance
3. **Question Format**: Be specific and clear in your questions
4. **Context**: Use follow-up questions to drill down into details
5. **Sources**: Always check sources to verify information
6. **Memory**: Clear chat when switching to different document sets

## 🔐 Security & Privacy

- ✅ All document processing happens locally
- ✅ API keys stored securely in `.env` (never committed to git)
- ✅ No data sent to third parties except LLM API calls
- ✅ Temporary files automatically cleaned up
- ✅ Vector store can be cleared anytime

## 📈 Performance Tips

- **Batch Processing**: Upload all documents at once for better efficiency
- **Optimal Chunk Size**: 1000 characters works well for most documents
- **Memory Window**: 5 previous conversations balances context and performance
- **Retrieval K**: 4 relevant chunks provides good coverage without noise

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional document formats (Markdown, HTML, etc.)
- More embedding models
- Advanced retrieval strategies
- UI enhancements
- Performance optimizations


## 🙏 Acknowledgments

Built with:
- [LangChain](https://langchain.com/)
- [Google Gemini](https://ai.google.dev/)
- [OpenAI](https://openai.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [HuggingFace](https://huggingface.co/)

