import streamlit as st
from pathlib import Path
from datetime import datetime
import json

# PDF utilities
from pypdf import PdfReader

# ==============================
# PROJECT IMPORTS (YOUR PIPELINE)
# ==============================
from ingestion.pdf_ingestion import ingest_pdf
from pipeline_incremental.pipeline_incremental import run_incremental_pipeline
from query.query_pipeline_v3 import process_user_query
from retrieval.retrieval_pipeline import retrieve_latest_query_chunks

# answer generation versions
from answer_generation.answer_generation_v2 import generate_answer_from_last_entry as generate_answer_v2
from answer_generation.answer_generation_v3 import generate_answer_from_last_entry as generate_answer_v3

# cleaner
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

if "multi_doc_confirmed" not in st.session_state:
    st.session_state.multi_doc_confirmed = False

if "answer_engine" not in st.session_state:
    st.session_state.answer_engine = "v3 (Grounded + Evidence)"


# =========================================================
# STARTUP CLEANING — DISABLED DURING TESTING
# =========================================================
# if "startup_clean_done" not in st.session_state:
#     clean_data_directories(PROJECT_ROOT_NEW)
#     st.session_state.startup_clean_done = True


# ==============================
# STYLING
# ==============================
st.markdown("""
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
""", unsafe_allow_html=True)


# ==============================
# HEADER
# ==============================
st.markdown('<div class="main-title">📚 RAG Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Upload PDF → Incremental Indexing → Chat with Knowledge</div>', unsafe_allow_html=True)
st.divider()


# ==============================
# HELPERS
# ==============================

def get_pdf_page_count(path: Path) -> int:
    return len(PdfReader(str(path)).pages)


def save_uploaded_file(uploaded_file) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = UPLOAD_DIR / f"{timestamp}_{uploaded_file.name.replace(' ', '_')}"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def load_existing_documents():
    docs = []
    if SOURCES_FILE.exists():
        with open(SOURCES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    uri = data.get("source_uri")
                    if uri:
                        filename = Path(uri).name
                        name = filename.replace(".pdf", "")
                        parts = name.split("_", 2)
                        if len(parts) == 3:
                            name = parts[2]
                        name = name.replace("_", " ")
                        docs.append(name)
                except Exception:
                    pass
    return sorted(set(docs))


def run_answer_generation():
    if st.session_state.answer_engine.startswith("v3"):
        return generate_answer_v3()
    else:
        return generate_answer_v2()

def render_evidence_metrics(payload: dict):

    metrics = payload.get("aggregate_evidence_metrics")
    if not metrics:
        return

    st.markdown("### 🔎 Evidence Quality")

    col1, col2, col3 = st.columns(3)

    # -------------------------
    # Citation coverage
    # -------------------------
    coverage = metrics.get("avg_citation_coverage")
    if coverage is not None:
        col1.metric("Citation Coverage", f"{coverage*100:.1f}%")
        st.progress(float(coverage))

    # -------------------------
    # Supported ratio
    # -------------------------
    supported = (
        metrics.get("total_supported_claims")
        or metrics.get("total_supported_sentences")
    )

    total = (
        metrics.get("total_claims")
        or metrics.get("total_sentences")
    )

    if supported is not None and total:
        ratio = supported / total
        col2.metric("Supported Evidence", f"{supported}/{total}")
        st.progress(float(ratio))

    # -------------------------
    # Confidence (optional)
    # -------------------------
    confidence = metrics.get("avg_evidence_confidence")
    if confidence is not None:
        col3.metric("Evidence Confidence", f"{confidence*100:.1f}%")

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("📥 Document Ingestion")


# ---------- Manual clean ----------
if st.sidebar.button("🧹 Clean All Data (Fresh Start)", use_container_width=True):
    clean_data_directories(PROJECT_ROOT_NEW)
    st.session_state.pipeline_ready = False
    st.session_state.chat_history = []
    st.session_state.multi_doc_confirmed = False
    st.success("All stored data cleared.")
    st.rerun()


st.sidebar.divider()


# ---------- Existing docs ----------
st.sidebar.subheader("📚 Existing Knowledge Sources")
existing_docs = load_existing_documents()

if existing_docs:
    for doc in existing_docs:
        st.sidebar.write(f"• {doc}")
else:
    st.sidebar.caption("No documents indexed yet")


st.sidebar.divider()


# ---------- Answer engine selector ----------
st.sidebar.subheader("🧠 Answer Generation Engine")
engine_choice = st.sidebar.radio(
    "Select version",
    ["v3 (Grounded + Evidence)", "v2 (Legacy)"],
    index=0 if st.session_state.answer_engine.startswith("v3") else 1
)
st.session_state.answer_engine = engine_choice


st.sidebar.divider()


# ---------- Upload ----------
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF (max 10 MB)",
    type=["pdf"],
    accept_multiple_files=True
)


if not uploaded_files:
    st.session_state.multi_doc_confirmed = False


# ==============================
# MULTI KNOWLEDGE WARNING
# ==============================
if uploaded_files:

    multiple_uploads = len(uploaded_files) > 1
    adding_to_existing = len(existing_docs) > 0
    knowledge_conflict = multiple_uploads or adding_to_existing

    if knowledge_conflict and not st.session_state.multi_doc_confirmed:

        st.sidebar.warning(
            "You are uploading documents that may belong to different knowledge domains. "
            "This can cause cross-referencing and inconsistent answers.\n\n"
            "Upload documents from the same knowledge domain for best results."
        )

        if st.sidebar.button("Yes, Continue Anyway"):
            st.session_state.multi_doc_confirmed = True
            st.rerun()

    else:
        if st.sidebar.button("🚀 Ingest & Process", use_container_width=True):
            try:
                processed_files = []

                for uploaded_file in uploaded_files:

                    size_mb = uploaded_file.size / (1024 * 1024)
                    if size_mb > MAX_FILE_SIZE_MB:
                        st.sidebar.error(f"{uploaded_file.name} exceeds 10 MB limit")
                        continue

                    saved_path = save_uploaded_file(uploaded_file)
                    pages = get_pdf_page_count(saved_path)

                    with st.spinner(f"Processing {uploaded_file.name} ({pages} pages)..."):
                        ingest_pdf(str(saved_path))
                        run_incremental_pipeline()

                    processed_files.append((uploaded_file.name, pages))

                if processed_files:
                    st.session_state.pipeline_ready = True
                    st.session_state.multi_doc_confirmed = False

                    summary = "<br>".join([f"📄 {name} — {pages} pages" for name, pages in processed_files])

                    st.sidebar.markdown(
                        f"""
                        <div class='success-box'>
                        ✅ Documents processed successfully<br>
                        {summary}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.rerun()

            except Exception as e:
                st.sidebar.error(f"Processing failed: {str(e)}")


# ==============================
# MAIN CHAT
# ==============================
st.header("💬 Chat with Your Documents")

if not st.session_state.pipeline_ready:
    st.info("Upload and process a document to start asking questions.")
else:

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask a question about your documents...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving knowledge and generating answer..."):
                try:
                    process_user_query(query)
                    retrieve_latest_query_chunks()
                    payload = run_answer_generation()

                    if isinstance(payload, dict):
                        answer = payload.get("answer") or str(payload)
                    else:
                        answer = str(payload)

                except Exception as e:
                    answer = f"Error generating answer: {str(e)}"

            st.markdown(answer)
            st.caption(f"Generated using {st.session_state.answer_engine}")

            # show metrics only for structured payload
            if isinstance(payload, dict):
                render_evidence_metrics(payload)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ==============================
# FOOTER
# ==============================
st.divider()
st.caption("Incremental RAG System • Streamlit UI")