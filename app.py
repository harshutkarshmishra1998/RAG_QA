import streamlit as st
from pathlib import Path
import time
from datetime import datetime
import json
import shutil

# PDF utilities
from pypdf import PdfReader

# ==============================
# PROJECT IMPORTS (YOUR PIPELINE)
# ==============================
from ingestion.pdf_ingestion import ingest_pdf
from pipeline_incremental.pipeline_incremental import run_incremental_pipeline
from query.query_pipeline_v3 import process_user_query
from retrieval.retrieval_pipeline import retrieve_latest_query_chunks
from answer_generation.answer_generation_v3 import generate_answer_from_last_entry

# NEW — CLEANER MODULE (create separately)
from patch.system_startup_cleaner import clean_data_directories


# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

PROJECT_ROOT_NEW = Path(__file__).resolve().parents[0]
UPLOAD_DIR = PROJECT_ROOT_NEW / "uploaded_files"
STORAGE_DIR = PROJECT_ROOT_NEW / "storage"
SOURCES_FILE = STORAGE_DIR / "sources.jsonl"

UPLOAD_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE_MB = 10


# ==============================
# SESSION STATE INIT
# ==============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

if "last_ingested_file" not in st.session_state:
    st.session_state.last_ingested_file = None

# ensures cleaning only happens once per app start (not per reload)
if "startup_clean_done" not in st.session_state:
    clean_data_directories(PROJECT_ROOT_NEW)
    st.session_state.startup_clean_done = True

# multi knowledge confirmation
if "multi_doc_confirmed" not in st.session_state:
    st.session_state.multi_doc_confirmed = False


# ==============================
# STYLING
# ==============================
st.markdown(
    """
    <style>
        .main-title { font-size: 32px; font-weight: 700; }
        .subtle { color: #888; font-size: 14px; }
        .success-box {
            padding: 12px;
            border-radius: 10px;
            background-color: #ecfdf5;
            border: 1px solid #10b981;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================
# HEADER
# ==============================
st.markdown('<div class="main-title">📚 RAG Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Upload PDF → Incremental Indexing → Chat with Knowledge</div>', unsafe_allow_html=True)
st.divider()


# ==============================
# HELPERS
# ==============================

def get_pdf_page_count(file_path: Path) -> int:
    reader = PdfReader(str(file_path))
    return len(reader.pages)


def estimate_processing_time_minutes(pages: int) -> float:
    return pages / 10


def save_uploaded_file(uploaded_file) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = uploaded_file.name.replace(" ", "_")
    target_path = UPLOAD_DIR / f"{timestamp}_{safe_name}"

    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return target_path


def load_existing_documents():
    docs = []
    if SOURCES_FILE.exists():
        with open(SOURCES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "source_name" in data:
                        docs.append(data["source_name"])
                except:
                    pass
    return list(sorted(set(docs)))


# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("📥 Document Ingestion")

# ---------- Manual Clean Button ----------
if st.sidebar.button("🧹 Clean All Data (Fresh Start)", use_container_width=True):
    clean_data_directories(PROJECT_ROOT_NEW)
    st.session_state.pipeline_ready = False
    st.session_state.chat_history = []
    st.success("All stored data cleared.")

st.sidebar.divider()

# ---------- Existing Docs ----------
st.sidebar.subheader("📚 Existing Knowledge Sources")
existing_docs = load_existing_documents()

if existing_docs:
    for doc in existing_docs:
        st.sidebar.write(f"• {doc}")
else:
    st.sidebar.caption("No documents indexed yet")

st.sidebar.divider()


# ---------- Upload ----------
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF (max 10 MB)",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------- Multi Knowledge Warning ----------
if uploaded_files:
    total_after_upload = len(existing_docs) + len(uploaded_files)

    if total_after_upload > 1 and not st.session_state.multi_doc_confirmed:
        st.sidebar.warning(
            "You are uploading multiple documents from different knowledge sources. "
            "This may lead to cross-referencing and potentially inconsistent or mixed answers. "
            "It is recommended to upload documents that belong to the same knowledge domain."
        )

        if st.sidebar.button("Yes, Continue Anyway"):
            st.session_state.multi_doc_confirmed = True
            st.rerun()

    else:
        if st.sidebar.button("🚀 Ingest & Process", use_container_width=True):
            try:
                for uploaded_file in uploaded_files:

                    file_size_mb = uploaded_file.size / (1024 * 1024)
                    if file_size_mb > MAX_FILE_SIZE_MB:
                        st.sidebar.error(f"{uploaded_file.name} exceeds 10 MB limit")
                        continue

                    saved_path = save_uploaded_file(uploaded_file)
                    pages = get_pdf_page_count(saved_path)
                    est_minutes = estimate_processing_time_minutes(pages)

                    with st.spinner(
                        f"Processing {uploaded_file.name}... Estimated ≤ {est_minutes:.1f} minutes"
                    ):
                        ingest_pdf(str(saved_path))
                        run_incremental_pipeline()

                st.session_state.pipeline_ready = True
                st.success("Documents processed successfully")

            except Exception as e:
                st.sidebar.error(f"Processing failed: {str(e)}")


# ==============================
# MAIN — CHAT QA
# ==============================
st.header("💬 Chat with Your Documents")

if not st.session_state.pipeline_ready:
    st.info("Upload and process a document to start asking questions.")
else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving knowledge and generating answer..."):
                try:
                    process_user_query(user_query)
                    retrieve_latest_query_chunks()
                    answer_payload = generate_answer_from_last_entry()

                    if isinstance(answer_payload, dict):
                        answer = (
                            answer_payload.get("answer")
                            or answer_payload.get("final_answer")
                            or answer_payload.get("generated_answer")
                            or str(answer_payload)
                        )
                    else:
                        answer = str(answer_payload)

                except Exception as e:
                    answer = f"Error generating answer: {str(e)}"

            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ==============================
# FOOTER
# ==============================
st.divider()
st.caption("Incremental RAG System • Streamlit UI")