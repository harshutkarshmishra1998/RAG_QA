import streamlit as st
from pathlib import Path
import time
from datetime import datetime

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


# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[0]
UPLOAD_DIR = PROJECT_ROOT / "uploaded_files"
UPLOAD_DIR.mkdir(exist_ok=True)

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


# ==============================
# STYLING
# ==============================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .subtle {
            color: #888;
            font-size: 14px;
        }
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


# ==============================
# SIDEBAR — INGESTION
# ==============================
st.sidebar.header("📥 Document Ingestion")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF (max 10 MB)",
    type=["pdf"]
)

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.sidebar.error(f"File size {file_size_mb:.2f} MB exceeds 10 MB limit.")
    else:
        if st.sidebar.button("🚀 Ingest & Process", use_container_width=True):

            try:
                saved_path = save_uploaded_file(uploaded_file)

                pages = get_pdf_page_count(saved_path)
                est_minutes = estimate_processing_time_minutes(pages)

                with st.spinner(
                    f"Processing document... Estimated time ≤ {est_minutes:.1f} minutes"
                ):
                    ingest_pdf(str(saved_path))
                    run_incremental_pipeline()
                    time.sleep(1)

                st.session_state.pipeline_ready = True
                st.session_state.last_ingested_file = saved_path.name

                st.sidebar.markdown(
                    f"""
                    <div class='success-box'>
                    ✅ Document processed successfully<br>
                    📄 Pages: {pages}<br>
                    📁 File: {saved_path.name}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.sidebar.error(f"Processing failed: {str(e)}")


# ==============================
# MAIN — CHAT QA
# ==============================
st.header("💬 Chat with Your Documents")

if not st.session_state.pipeline_ready:
    st.info("Upload and process a document to start asking questions.")
else:
    if st.session_state.last_ingested_file:
        st.caption(f"Active knowledge base: {st.session_state.last_ingested_file}")

    # Display history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        # ==============================
        # SHOW USER MESSAGE
        # ==============================
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })

        with st.chat_message("user"):
            st.markdown(user_query)

        # ==============================
        # RAG PIPELINE EXECUTION
        # ==============================
        with st.chat_message("assistant"):
            with st.spinner("Retrieving knowledge and generating answer..."):
                try:
                    # STEP 1 — QUERY PROCESSING
                    process_user_query(user_query)

                    # STEP 2 — RETRIEVE LATEST QUERY CHUNKS
                    retrieve_latest_query_chunks()

                    # STEP 3 — ANSWER GENERATION
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

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })


# ==============================
# FOOTER
# ==============================
st.divider()
st.caption("Incremental RAG System • Streamlit UI")