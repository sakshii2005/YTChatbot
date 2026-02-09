# 🎬 YouTube Transcript QA with RAG and HuggingFace (Streamlit App)

This Streamlit app allows users to input a **YouTube video ID** and a **question** to get intelligent answers based on the video transcript using a **Retrieval-Augmented Generation (RAG)** pipeline.

## 🚀 Features

- 🔍 Fetches the English transcript of any YouTube video.
- 📚 Splits and vectorizes the transcript using `sentence-transformers/all-mpnet-base-v2`.
- 🧠 Uses `FAISS` for fast similarity search over transcript chunks.
- 💬 Powered by `microsoft/Phi-3-mini-4k-instruct` from Hugging Face for accurate and lightweight reasoning.
- 🛠️ Built using `LangChain`, `HuggingFace`, and `Streamlit`.

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/youtube-rag-app.git
cd youtube-rag-app
