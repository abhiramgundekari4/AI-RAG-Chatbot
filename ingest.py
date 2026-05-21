import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH   = "data/"
DB_FAISS_PATH = "faiss_index"

def create_vector_store():
    print("Loading documents from:", DATA_PATH)

    # Load all PDFs inside data/ folder
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"  Loaded {len(documents)} pages")

    if not documents:
        print("ERROR: No PDFs found in data/ folder!")
        print("Add at least one PDF file to the data/ folder and try again.")
        return
 # Split pages into smaller chunks
    # chunk_size=500: each chunk is ~500 characters
    # chunk_overlap=50: 50 chars overlap so context is not lost
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks")

    # Load free embedding model (downloads once, ~90MB)
    # This converts text to vectors (numbers) for similarity search
    print("Loading embedding model (first time takes ~1 min to download)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Create FAISS vector store from chunks
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save to disk
    vectorstore.save_local(DB_FAISS_PATH)
    print(f"\nDone! Vector store saved to '{DB_FAISS_PATH}/'")
    print("Now run: streamlit run app.py")

if __name__ == "__main__":
    create_vector_store()
