echo # Architecture Overview> architecture.md
echo The RAG chatbot consists of:>> architecture.md
echo - **LLM**: google/flan-t5-base for response generation.>> architecture.md
echo - **Vector Database**: FAISS with sentence-transformers/all-mpnet-base-v2, k=7.>> architecture.md
echo - **Retrieval Chain**: LangChain RetrievalQA for context retrieval and answer generation.>> architecture.md
echo - **API**: FastAPI /chat endpoint with answers and source filenames.>> architecture.md
echo - **Document Loader**: PyMuPDFLoader for sample.pdf, chunk_size=1000, chunk_overlap=100.>> architecture.md
