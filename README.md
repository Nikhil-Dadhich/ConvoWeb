# 🤖 ConvoWeb

**ConvoWeb** is a Conversational RAG (Retrieval-Augmented Generation) application built with **Streamlit**, **LangChain**, **Google Gemini**, and **Chroma Vector Store**. It allows users to ask questions based on the contents of a given website by loading and indexing its content, then using a history-aware conversational interface to provide accurate, context-rich responses.

---

## 🧠 Features

- ✅ Load and parse webpage content using `WebBaseLoader`
- ✅ Split content into manageable chunks with `RecursiveCharacterTextSplitter`
- ✅ Embed content using `GoogleGenerativeAIEmbeddings`
- ✅ Store and retrieve chunks with `Chroma` vector store
- ✅ Use LangChain's `create_history_aware_retriever` for contextual search
- ✅ Gemini-powered response generation via `ChatGoogleGenerativeAI`
- ✅ Chat memory via `st.session_state` for multi-turn conversations
- ✅ Simple and intuitive Streamlit UI

---

## 📷 Design Overview

You can include the following images for better clarity on how the system works:

 **RAG Pipeline Diagram**
 ![RAG Pipeline](https://github.com/Nikhil-Dadhich/ConvoWeb/blob/main/RAG.png)

---

## 🚀 How to Run

### 🔧 Requirements

- Python 3.9+
- Packages listed in `requirements.txt` (see below)
- Google Gemini API Key in a `.env` file

### 🧪 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Nikhil-Dadhich/ConvoWeb.git
cd ConvoWeb

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Add your Google API key to .env
echo "GEMINI_API_KEY=your_google_gemini_key" > .env

# Run the Streamlit app
streamlit run app.py
```

---

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_google_gemini_api_key
```

---

## 🗣️ Example Usage

1. Enter a valid website URL in the sidebar.
2. Wait while the content is parsed and vectorized.
3. Ask questions in the chat interface based on the content.
4. Enjoy rich, contextual answers from the chatbot!

---

## 🧩 Components

- **`get_vectorstore()`**: Loads and splits the website content and stores in Chroma.
- **`get_context_retriever_chain()`**: Creates a retriever chain using chat history.
- **`get_conversational_rag_chain()`**: Combines retriever and LLM for RAG.
- **Streamlit App**: Handles user input and manages chat state.

---

## 👨‍💻 Author

**Nikhil CR7**

Built with ❤️ using Streamlit and LangChain.

---