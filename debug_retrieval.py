echo import logging> debug_retrieval.py
echo from langchain_community.document_loaders import PyMuPDFLoader>> debug_retrieval.py
echo from langchain.text_splitter import RecursiveCharacterTextSplitter>> debug_retrieval.py
echo from langchain_community.vectorstores import FAISS>> debug_retrieval.py
echo from langchain_huggingface import HuggingFaceEmbeddings>> debug_retrieval.py
echo.>> debug_retrieval.py
echo # Set up logging>> debug_retrieval.py
echo logging.basicConfig(level=logging.INFO, format='%%(asctime)s - %%(levelname)s - %%(message)s')>> debug_retrieval.py
echo logger = logging.getLogger(__name__)>> debug_retrieval.py
echo.>> debug_retrieval.py
echo def load_and_index_documents():>> debug_retrieval.py
echo     try:>> debug_retrieval.py
echo         pdf_path = "sample.pdf">> debug_retrieval.py
echo         logger.info(f"Loading PDF from {pdf_path}...")>> debug_retrieval.py
echo         loader = PyMuPDFLoader(pdf_path)>> debug_retrieval.py
echo         documents = loader.load()>> debug_retrieval.py
echo         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)>> debug_retrieval.py
echo         texts = splitter.split_documents(documents)>> debug_retrieval.py
echo         logger.info(f"Split into {len(texts)} chunks")>> debug_retrieval.py
echo         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")>> debug_retrieval.py
echo         vector_store = FAISS.from_documents(texts, embeddings)>> debug_retrieval.py
echo         return vector_store>> debug_retrieval.py
echo     except Exception as e:>> debug_retrieval.py
echo         logger.error(f"Error: {str(e)}")>> debug_retrieval.py
echo         raise>> debug_retrieval.py
echo.>> debug_retrieval.py
echo def debug_retrieval():>> debug_retrieval.py
echo     vector_store = load_and_index_documents()>> debug_retrieval.py
echo     retriever = vector_store.as_retriever(search_kwargs={"k": 7})>> debug_retrieval.py
echo     question = "How is FAISS used in the chatbot?">> debug_retrieval.py
echo     docs = retriever.invoke(question)>> debug_retrieval.py
echo     logger.info(f"Retrieved {len(docs)} documents for question: {question}")>> debug_retrieval.py
echo     for i, doc in enumerate(docs):>> debug_retrieval.py
echo         logger.info(f"\nDoc {i+1}:\n{doc.page_content[:200]}...")>> debug_retrieval.py
echo         logger.info(f"Metadata: {doc.metadata}")>> debug_retrieval.py
echo.>> debug_retrieval.py
echo if __name__ == "__main__":>> debug_retrieval.py
echo     debug_retrieval()>> debug_retrieval.py