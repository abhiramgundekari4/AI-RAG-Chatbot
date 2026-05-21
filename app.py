import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }

[data-testid="stSidebar"] { background: #0d0d18; border-right: 1px solid #1a1a30; }
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem 2rem; max-width: 860px; margin: 0 auto; }

/* Hero */
.hero { text-align: center; padding: 2.5rem 0 1.5rem 0; }
.hero h1 {
    font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; letter-spacing: -1px;
}
.hero p { color: #6b7280; font-size: 1rem; margin-top: 6px; font-weight: 300; }

/* Status */
.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 99px; font-size: 12px; font-weight: 500; margin: 4px 0;
}
.status-ready   { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.status-waiting { background: #1c1917; color: #a8a29e; border: 1px solid #292524; }

/* Messages */
.msg-user {
    background: linear-gradient(135deg, #1e1b4b, #1a1a35);
    border: 1px solid #3730a3; border-radius: 16px 16px 4px 16px;
    padding: 12px 18px; margin: 6px 0; margin-left: 8%;
    color: #c7d2fe; font-size: 0.95rem; line-height: 1.6;
}
.msg-bot {
    background: #0f0f1a; border: 1px solid #1e1e35;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 18px; margin: 6px 0; margin-right: 8%;
    color: #e8e8f0; font-size: 0.95rem; line-height: 1.6;
}
.msg-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 5px; opacity: 0.45;
}

/* Source */
.source-pill {
    display: inline-block; background: #1a1a2e; border: 1px solid #2d2d4e;
    border-radius: 6px; padding: 3px 10px; font-size: 11px;
    color: #818cf8; margin: 3px 3px 3px 0;
}

/* Sidebar labels & rows */
.sidebar-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #9ca3af; margin: 14px 0 8px 0;
}
.sidebar-row {
    font-size: 13px; color: #9ca3af; padding: 3px 0;
    display: flex; align-items: center; gap: 8px; line-height: 1.5;
}

/* Stat cards */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 4px; }
.stat-card {
    background: #13131f; border: 1px solid #1e1e35; border-radius: 10px;
    padding: 10px 12px; text-align: center;
}
.stat-val { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700; color: #a78bfa; }
.stat-lbl { font-size: 10px; color: #4b5563; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

/* Tips */
.tip-card {
    background: #0f0f1a; border: 1px solid #1e1e35; border-radius: 10px;
    padding: 10px 12px; margin-bottom: 6px;
}
.tip-card p { font-size: 12px; color: #9ca3af; margin: 0; line-height: 1.5; }
.tip-card span { color: #a78bfa; font-weight: 600; }

/* Builder card */
.builder-card {
    background: #13131f; border: 1px solid #2d2d4e;
    border-radius: 14px; padding: 14px 16px; margin-top: 8px;
}
.builder-name {
    font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.builder-info { font-size: 12px; color: #6b7280; margin-top: 3px; line-height: 1.6; }
.builder-links { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.builder-link {
    font-size: 11px; color: #818cf8 !important; text-decoration: none !important;
    background: #1a1a2e; border: 1px solid #2d2d4e; border-radius: 6px; padding: 3px 10px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0f0f1a; border: 2px dashed #2d2d4e;
    border-radius: 16px; padding: 1rem;
}
[data-testid="stFileUploader"]:hover { border-color: #a78bfa; }
[data-testid="stFileUploader"] label { color: #9ca3af !important; }

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #0f0f1a !important; border: 1px solid #2d2d4e !important;
    border-radius: 12px !important; color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.12) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    color: white !important; border: none; border-radius: 10px;
    font-family: 'DM Sans', sans-serif; font-weight: 500;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

/* Warning / info */
[data-testid="stAlert"] {
    background: #1a1207 !important; border: 1px solid #854d0e !important;
    border-radius: 10px !important; color: #fcd34d !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #0f0f1a; border: 1px solid #1e1e35 !important; border-radius: 10px;
}
hr { border-color: #1e1e35; }
</style>
""", unsafe_allow_html=True)

# ── Prompt ────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are DocMind, a precise document assistant. Answer using ONLY the context below.
If the answer is not in the context, say: "I couldn't find that in the uploaded document."
Never use outside knowledge. Be clear and concise.

Context:
{context}

Question: {question}

Answer:"""

# ── Helpers ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def process_uploaded_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs   = loader.load()
    os.unlink(tmp_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
    )
    chunks     = splitter.split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", temperature=0, max_tokens=1024,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    chain  = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        return_source_documents=True, chain_type_kwargs={"prompt": prompt}
    )
    return chain, len(chunks), len(docs)

def reset_session():
    for k in ["messages", "chain", "doc_name", "doc_pages",
              "doc_chunks", "confirm_back", "total_questions"]:
        default = [] if k == "messages" else (False if k == "confirm_back" else
                  0 if k in ["doc_pages","doc_chunks","total_questions"] else None)
        st.session_state[k] = default

# ── Session state ─────────────────────────────────────────────
defaults = {
    "messages": [], "chain": None, "doc_name": None,
    "doc_pages": 0, "doc_chunks": 0,
    "confirm_back": False, "total_questions": 0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
    background:linear-gradient(135deg,#a78bfa,#60a5fa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;padding:1.2rem 0 0.3rem 0;">🧠 DocMind AI</div>
    """, unsafe_allow_html=True)

    # Status
    if st.session_state.chain:
        st.markdown(f'<div class="status-badge status-ready">✦ {st.session_state.doc_name}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-waiting">○ No document loaded</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 1: Session stats ──
    st.markdown('<div class="sidebar-label">📊 Session Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-val">{st.session_state.doc_pages}</div>
            <div class="stat-lbl">Pages</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{st.session_state.doc_chunks}</div>
            <div class="stat-lbl">Chunks</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{st.session_state.total_questions}</div>
            <div class="stat-lbl">Questions</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{'✓' if st.session_state.chain else '–'}</div>
            <div class="stat-lbl">AI Ready</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 2: How it works ──
    st.markdown('<div class="sidebar-label">🚀 How It Works</div>', unsafe_allow_html=True)
    for icon, text in [("①", "Upload any PDF"), ("②", "AI indexes it instantly"),
                       ("③", "Ask anything about it"), ("④", "See source chunks used")]:
        st.markdown(f'<div class="sidebar-row"><span style="color:#a78bfa;font-weight:600">{icon} </span>{text}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 3: Tips ──
    st.markdown('<div class="sidebar-label">💡 Pro Tips</div>', unsafe_allow_html=True)
    tips = [
        ("Ask specific questions", "e.g. 'What is the main conclusion of chapter 3?'"),
        ("Request summaries", "e.g. 'Summarize the introduction'"),
        ("Compare sections", "e.g. 'What's the difference between X and Y?'"),
    ]
    for title, example in tips:
        st.markdown(f"""
        <div class="tip-card">
            <p><span>{title}</span><br>{example}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 4: Powered by ──
    st.markdown('<div class="sidebar-label">⚙️ Powered By</div>', unsafe_allow_html=True)
    for icon, tech in [("⚡", "Groq LLaMA 3.3 70B"), ("🔍", "FAISS Vector Search"),
                       ("🤗", "HuggingFace Embeddings"), ("🦜", "LangChain RAG")]:
        st.markdown(f'<div class="sidebar-row">{icon} {tech}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 5: Actions ──
    st.markdown('<div class="sidebar-label">🛠 Actions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_questions = 0
            st.rerun()
    with col2:
        if st.button("📄 New doc", use_container_width=True):
            reset_session()
            st.rerun()

    st.markdown("---")

    # ── Section 6: Built by ──
    # 🔑 UPDATE YOUR NAME, DEGREE, GITHUB & LINKEDIN BELOW
    st.markdown('<div class="sidebar-label">👨‍💻 Built By</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="builder-card">
        <div class="builder-name">Abhiram</div>
        <div class="builder-info">
            B.E. Computer Science & Engineering<br>
            Final Year · 2025
        </div>
        <div class="builder-links">
            <a class="builder-link" href="https://github.com/yourusername" target="_blank">GitHub ↗</a>
            <a class="builder-link" href="https://linkedin.com/in/yourprofile" target="_blank">LinkedIn ↗</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>DocMind AI</h1>
    <p>Upload a PDF · Ask anything · Built by
    <strong style="color:#a78bfa;font-style:normal;">Abhiram</strong></p>
</div>
""", unsafe_allow_html=True)

# ── Upload screen ─────────────────────────────────────────────
if st.session_state.chain is None:
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    st.markdown("""
    <div style="text-align:center;color:#374151;font-size:13px;margin-top:10px;">
        📎 Supports any PDF — textbook, notes, report, research paper
    </div>
    """, unsafe_allow_html=True)

    if uploaded:
        with st.spinner(f"📖 Indexing **{uploaded.name}**..."):
            chain, n_chunks, n_pages = process_uploaded_pdf(uploaded)
            st.session_state.chain       = chain
            st.session_state.doc_name    = uploaded.name
            st.session_state.doc_pages   = n_pages
            st.session_state.doc_chunks  = n_chunks
            st.session_state.messages    = [{
                "role": "assistant",
                "content": f"✅ **{uploaded.name}** is ready!\n\n📄 {n_pages} pages indexed into {n_chunks} chunks.\nAsk me anything about this document.",
                "sources": []
            }]
        st.rerun()

# ── Chat screen ───────────────────────────────────────────────
else:
    # ── Top bar ──
    col_doc, col_btn = st.columns([3, 1])
    with col_doc:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:8px 0 4px 0;">
            <span style="font-size:18px;">📄</span>
            <span style="font-size:13px;color:#6b7280;">Chatting with</span>
            <span style="font-size:13px;color:#a78bfa;font-weight:500;">{st.session_state.doc_name}</span>
            <span style="font-size:11px;color:#374151;">· {st.session_state.doc_pages} pages</span>
        </div>
        """, unsafe_allow_html=True)
    with col_btn:
        if st.button("⬅ Upload new PDF", use_container_width=True):
            st.session_state.confirm_back = True

    # ── Confirmation dialog ──
    if st.session_state.get("confirm_back"):
        st.warning("⚠️ Going back will clear your current chat. Are you sure?")
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            if st.button("✅ Yes, go back", use_container_width=True):
                st.session_state.confirm_back = False
                reset_session()
                st.rerun()
        with c2:
            if st.button("❌ No, stay", use_container_width=True):
                st.session_state.confirm_back = False
                st.rerun()

    st.markdown('<hr style="margin:4px 0 16px 0;">', unsafe_allow_html=True)

    # ── Messages ──
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
                <div class="msg-label">You</div>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            content = msg["content"].replace("\n", "<br>")
            st.markdown(f"""
            <div class="msg-bot">
                <div class="msg-label">🧠 DocMind</div>
                {content}
            </div>
            """, unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander(f"📎 View {len(msg['sources'])} source chunks", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f'<span class="source-pill">Chunk {i} · Page {src["page"]}</span>', unsafe_allow_html=True)
                        st.markdown(f'<div style="font-size:12px;color:#6b7280;padding:4px 0 12px 4px;line-height:1.6;">{src["text"][:280]}...</div>', unsafe_allow_html=True)

    # ── Chat input ──
    if user_input := st.chat_input(f"Ask about {st.session_state.doc_name}..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.total_questions += 1

        with st.spinner("🔍 Searching document..."):
            result      = st.session_state.chain.invoke({"query": user_input})
            answer      = result["result"]
            source_data = [
                {"text": doc.page_content, "page": doc.metadata.get("page", "?")}
                for doc in result["source_documents"]
            ]

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": source_data
        })
        st.rerun()

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 0.5rem 0;color:#1f2937;font-size:12px;">
    Built by <span style="color:#a78bfa;font-weight:500;">Abhiram</span> ·
    Powered by LangChain · Groq · FAISS
</div>
""", unsafe_allow_html=True)