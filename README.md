# 🔍 Hybrid Search with RAG-Powered Q&A System

This project combines the 🌀 power of document retrieval with advanced 🧠 language model capabilities to deliver ✅ precise, 🔄 contextual responses. Using 🔎 Hybrid Search and Retrieval Augmented Generation (RAG), it bridges the gap between document-specific knowledge and general 🧠 intelligence.

---

## 📝 Features

- **📃 PDF Uploads**: Allows users to upload and process 📔 PDF files.
- **✍️ Text Chunking and Embedding**: Automatically generates searchable 🔎 text chunks.
- **🔍 Hybrid Search**: Combines 🎨 semantic and keyword-based search techniques.
- **🚀 Intelligent Responses**: Delivers high-quality answers using Claude’s 🧠 language capabilities.
- **🔄 Fallback Mechanism**: Defaults to 🔐 general knowledge when document-specific answers are unavailable.
- **📣️ Interactive Chat**: Provides an intuitive and user-friendly interface.

---

## 🔎 What is Hybrid Search RAG?

Hybrid Search RAG blends two powerful techniques to ensure ✔️ comprehensive and ✅ accurate information retrieval:

1. **🎨 Semantic Search**: Uses 🧠 embeddings to identify contextually similar information.
2. **🔎 Keyword Search**: Finds ✅ exact or closely related matches to specific terms.

RAG (Retrieval Augmented Generation) leverages retrieved 🔎 content to generate responses grounded in the uploaded documents, ensuring relevance and precision.

---

## 🔧 Core Components

### 1. RAGLite

- 🔄 Python toolkit for RAG operations.
- 🔢 Handles document processing, chunking, and embedding.
- 🔍 Supports both vector and keyword searches.

### 2. Model Stack

- **Claude 3 Opus**: Primary 🧠 language model for response generation.
- **OpenAI text-embedding-3-large**: Creates 🧠 semantic embeddings for searches.
- **Cohere Reranker**: Reorders search results for improved relevance.

### 3. 📂 Database Options

- **PostgreSQL**: Recommended for production environments.
- **SQLite**: Suitable for development purposes.

---

## ℹ️ Prerequisites

Before setting up the application, ensure you have the following:

### 1. 📂 Database

- Create a free PostgreSQL database using 📝 Neon:
  - ✅ Sign up or log in to Neon.
  - ⚖️ Create a new project.
  - 🔐 Copy the connection string (e.g., `postgresql://user:pass@ep-xyz.region.aws.neon.tech/dbname`).

### 2. 🔑 API Keys

- **OpenAI API key**: For embeddings.
- **Anthropic API key**: For Claude.
- **Cohere API key**: For reranking.

### 3. 📝 Software

- **Python**: Version 3.7 or higher.
- **Code Editor**: VS Code or PyCharm recommended.
- **🔧 Basic Knowledge**: Familiarity with Python programming.

---

## 📚 Installation and Setup

### 1. 🔍 Clone the Repository

```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. 🔑 Configure API Keys and Database

Add your API keys and database URL to `my_keys.py`:

```python
# my_keys.py
OPENAI_API_KEY = "your_openai_key"
ANTHROPIC_API_KEY = "your_anthropic_key"
COHERE_API_KEY = "your_cohere_key"
DATABASE_URL = "your_database_url"
```

### 3. 🔧 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. 🚀 Run the Application

```bash
streamlit run app.py
```

---

## 🔎 How to Use

### 1. 📃 Upload Documents

- Use the file uploader in the application to upload 📔 PDF files for processing.

### 2. 🧠 Ask Questions

- Once documents are processed, use the 📣️ chat interface to ask questions.
- If no document-specific answer is found, the system will fallback to using general AI knowledge.

### 3. 🔎 View Responses

- Get ✅ precise and 🔄 contextual answers directly in the 📣️ chat interface.

---

## 🛠️ Contributing

🖐 Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

---

## 🔒 License

This project is licensed under the © MIT License.

