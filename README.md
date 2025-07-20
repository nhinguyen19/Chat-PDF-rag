# Chat-PDF-rag
Built a Streamlit chatbot that answers questions from uploaded PDFs using Google Gemini embeddings and FAISS for semantic search.
# 🛠️ Tools
Python, Streamlit, LangChain, Google Generative AI (Gemini), FAISS, PDF Parsing

# 📄 Summary
🧠 Built a Streamlit-based conversational AI chatbot that answers user questions from uploaded PDF documents.

🔍 Google Gemini embedding model (embedding-001) used for generating vector representations of text.

⚡ Used FAISS for semantic vector search and efficient retrieval of relevant content.

📚 Employed LangChain's RecursiveCharacterTextSplitter to divide PDF text into meaningful chunks.

🧾 Supported multi-PDF upload, processed with tempfile and PyPDFLoader.

🤖 Designed custom prompt template to ensure answers are grounded strictly in provided context (reduce hallucinations).

🔄 Used asyncio to resolve Streamlit event loop conflicts during vector store creation.

🚨 Added robust exception handling and user feedback in UI when errors occur (e.g., missing API key, embedding errors).

