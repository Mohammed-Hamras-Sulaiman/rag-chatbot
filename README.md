
# 🧠 RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot developed for the RAG Chatbot Assessment.

## 🚀 Features

- 🔍 Retrieves relevant context from PDFs using FAISS (with `k=7` neighbors).
- 🤖 Generates answers using `google/flan-t5-base` via LangChain's `RetrievalQA`.
- 🌐 FastAPI `/chat` endpoint with optional source tracking.
- 🧪 Includes a debug script to test document retrieval.

## 📦 Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/<your_username>/rag-chatbot.git
   cd rag-chatbot
````

2. **Create & Activate a Virtual Environment**

   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Model (FLAN-T5)**

   ```bash
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='google/flan-t5-base', local_dir='flan-t5-base')"
   ```

5. **Add your document**

   Place a PDF file named `sample.pdf` in the project directory.

6. **Run the FastAPI Server**

   ```bash
   uvicorn rag_chat_api:app --reload
   ```

7. **(Optional) Debug Retrieval**

   ```bash
   python debug_retrieval.py
   ```

## 🔗 API Usage

* **Endpoint:** `POST /chat`

* **Request Body:**

  ```json
  {
    "question": "What is FAISS?",
    "session_id": "test_session"
  }
  ```

* **Sample Response:**

  ```json
  {
    "answer": "FAISS is a library for efficient similarity search...",
    "sources": [
      {
        "content": "...",
        "filename": "sample.pdf"
      }
    ]
  }
  ```

* **Example cURL:**

  ```bash
  curl -X POST http://127.0.0.1:8000/chat \
       -H "Content-Type: application/json" \
       -d "{\"question\": \"What is FAISS?\", \"session_id\": \"test_session\"}"
  ```

* **Interactive Docs (Swagger UI):**

  [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 📁 Project Structure

```
rag-chatbot/
├── rag_chat_api.py        # FastAPI server entrypoint
├── rag_chatbot.py         # LangChain-based RAG pipeline
├── debug_retrieval.py     # Script to test/document retrieval
├── sample.pdf             # Sample document (user-provided)
├── requirements.txt       # Python dependencies
├── architecture.md        # Architecture overview (optional)
└── README.md              # This file
```

## 🛠️ Technologies Used

* LangChain
* Hugging Face Transformers
* Sentence Transformers
* FAISS (Facebook AI Similarity Search)
* FastAPI
* PyMuPDF

## 📌 Notes

* Set `k` in the FAISS retriever to control how many chunks to retrieve.
* This project is **local only** and does not require internet after model is downloaded.

---

Feel free to contribute or raise issues if you're improving or testing the RAG chatbot. Happy building! 🚀

```

---

Let me know if you want a `LICENSE` file, `.gitignore`, or GitHub Actions for deployment/testing too!
```
