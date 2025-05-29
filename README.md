echo # RAG Chatbot> README.md
echo A Retrieval-Augmented Generation chatbot for the RAG Chatbot Assessment.>> README.md
echo ## Features>> README.md
echo - Retrieves context from PDFs using FAISS (k=7).>> README.md
echo - Generates answers with google/flan-t5-base via LangChain RetrievalQA.>> README.md
echo - FastAPI /chat endpoint with source tracking.>> README.md
echo - Debug script to verify retrieval.>> README.md
echo ## Setup>> README.md
echo 1. Clone: `git clone https://github.com/<your_username>/rag-chatbot.git`>> README.md
echo 2. Activate venv: `.\venv\Scripts\activate`>> README.md
echo 3. Install: `pip install -r requirements.txt`>> README.md
echo 4. Download model: `python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='google/flan-t5-base', local_dir='flan-t5-base')"`>> README.md
echo 5. Add `sample.pdf`.>> README.md
echo 6. Run: `uvicorn rag_chat_api:app`>> README.md
echo 7. Debug retrieval: `python debug_retrieval.py`>> README.md
echo ## API Usage>> README.md
echo - **Endpoint**: `POST /chat`>> README.md
echo - **Request**: `{"question": "What is FAISS?", "session_id": "test_session"}`>> README.md
echo - **Response**: `{"answer": "...", "sources": [{"content": "...", "filename": "sample.pdf"}]}`>> README.md
echo - **Example**: `curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What is FAISS?\", \"session_id\": \"test_session\"}"`>> README.md
echo - **Swagger UI**: `http://127.0.0.1:8000/docs`>> README.md
echo ## Structure>> README.md
echo - `rag_chat_api.py`: FastAPI server>> README.md
echo - `rag_chatbot.py`: Reference pipeline>> README.md
echo - `debug_retrieval.py`: Retrieval debug script>> README.md
echo - `sample.pdf`: Input document>> README.md
echo - `architecture.md`: Architecture overview>> README.md
echo - `requirements.txt`: Dependencies>> README.md
