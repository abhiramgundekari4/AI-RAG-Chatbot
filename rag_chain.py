import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

DB_FAISS_PATH = "faiss_index"

PROMPT_TEMPLATE = """You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not found in the context, say:
"I don't have enough information in the documents to answer this."
Do not use any outside knowledge.

Context:
{context}

Question: {question}

Answer:"""

def load_rag_chain():
    """Load vector store and build the RAG chain. Call once at startup."""

    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # ✅ Updated model name — llama3-8b-8192 is decommissioned
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("RAG chain ready!")
    return qa_chain


def ask(chain, question: str):
    """Ask a question and return answer + source chunks."""
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]
    return answer, sources


if __name__ == "__main__":
    chain = load_rag_chain()
    test_q = "What is this document about?"
    answer, sources = ask(chain, test_q)
    print(f"\nQ: {test_q}")
    print(f"A: {answer}")
    print(f"\nSources used: {len(sources)} chunks")
    for i, doc in enumerate(sources, 1):
        print(f"  [{i}] {doc.page_content[:100]}...")