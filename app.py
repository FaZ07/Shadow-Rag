"""
Shadow RAG — Streamlit application entry point.
Run: streamlit run app.py
"""
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# ── Page config must be first Streamlit call ─────────────────────────────────
st.set_page_config(
    page_title="Shadow RAG",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded",
)

from rag.embedder import EmbeddingStore
from rag.generator import LLMClient, generate_answer
from rag.ingest import load_documents, semantic_chunk
from rag.retriever import Retriever
from rag.shadow import shadow_critique
from rag.verifier import compute_confidence_score, verify_all_claims

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
INDEX_DIR = Path("index")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #6b7280; font-size: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border: 1px solid #2d2d3f;
    }
    .score-high { color: #10b981; font-weight: 700; font-size: 2rem; }
    .score-mid  { color: #f59e0b; font-weight: 700; font-size: 2rem; }
    .score-low  { color: #ef4444; font-weight: 700; font-size: 2rem; }
    .tag-verified   { background: #065f46; color: #6ee7b7; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .tag-partial    { background: #78350f; color: #fcd34d; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .tag-unsupported{ background: #7f1d1d; color: #fca5a5; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .risk-low    { color: #10b981; font-weight: 600; }
    .risk-medium { color: #f59e0b; font-weight: 600; }
    .risk-high   { color: #ef4444; font-weight: 600; }
    div[data-testid="stExpander"] { border: 1px solid #2d2d3f; border-radius: 8px; }
    .stButton > button { border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "store": None,
        "retriever": None,
        "llm": None,
        "index_built": False,
        "doc_names": [],
        "last_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Cached initialisation helpers ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedding_store() -> EmbeddingStore:
    store = EmbeddingStore()
    # Try loading persisted index
    if store.load(INDEX_DIR / "faiss.index", INDEX_DIR / "metadata.json"):
        logger.info("Loaded existing FAISS index.")
    return store


@st.cache_resource(show_spinner="Initialising LLM…")
def get_llm() -> Optional[LLMClient]:
    try:
        return LLMClient()
    except (EnvironmentError, ImportError) as exc:
        st.error(f"LLM init failed: {exc}")
        return None


def build_index_from_data(store: EmbeddingStore) -> List[str]:
    """Ingest /data folder and build FAISS index."""
    docs = load_documents(str(DATA_DIR))
    if not docs:
        return []
    chunks = semantic_chunk(docs)
    store.build_index(chunks)
    store.save(INDEX_DIR / "faiss.index", INDEX_DIR / "metadata.json")
    return list({d["source"] for d in docs})


def save_uploaded_file(uploaded_file) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / uploaded_file.name
    dest.write_bytes(uploaded_file.read())
    return dest


# ── Confidence score colour helper ───────────────────────────────────────────
def _score_class(score: int) -> str:
    if score >= 70:
        return "score-high"
    if score >= 40:
        return "score-mid"
    return "score-low"


def _status_badge(status: str) -> str:
    mapping = {
        "VERIFIED": '<span class="tag-verified">✓ VERIFIED</span>',
        "PARTIALLY_SUPPORTED": '<span class="tag-partial">~ PARTIAL</span>',
        "NOT_SUPPORTED": '<span class="tag-unsupported">✗ NOT SUPPORTED</span>',
    }
    return mapping.get(status, status)


def _risk_badge(risk: str) -> str:
    cls = f"risk-{risk}" if risk in ("low", "medium", "high") else ""
    icons = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    icon = icons.get(risk, "⚪")
    return f'<span class="{cls}">{icon} {risk.upper()}</span>'


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(
    question: str,
    retriever: Retriever,
    llm: LLMClient,
    shadow_mode: bool,
    fact_check_mode: bool,
    top_k: int,
) -> Dict:
    # 1. Retrieve
    chunks = retriever.retrieve(question, top_k=top_k)
    context = retriever.format_context(chunks)
    sources = list({c["source"] for c in chunks})

    # 2. Generate answer
    answer = generate_answer(question, context, llm)

    result: Dict = {
        "question": question,
        "answer": answer,
        "confidence_score": None,
        "shadow_analysis": None,
        "fact_check": [],
        "sources": sources,
    }

    # 3. Shadow critique
    if shadow_mode:
        result["shadow_analysis"] = shadow_critique(question, answer, context, llm)

    # 4. Claim verification
    if fact_check_mode:
        result["fact_check"] = verify_all_claims(answer, retriever, question, llm, top_k=top_k)

    # 5. Confidence score
    shadow = result["shadow_analysis"] or {"hallucination_risk": "unknown", "contradictions": []}
    result["confidence_score"] = compute_confidence_score(result["fact_check"], shadow)

    return result


# ── Render result ─────────────────────────────────────────────────────────────
def render_result(result: Dict, shadow_mode: bool, fact_check_mode: bool):
    st.markdown("---")

    # Row: answer + score side-by-side
    col_ans, col_score = st.columns([3, 1])

    with col_ans:
        st.subheader("🧠 Answer")
        st.markdown(result["answer"])

    with col_score:
        score = result["confidence_score"]
        css_class = _score_class(score)
        st.markdown("**📊 Confidence Score**")
        st.markdown(f'<div class="{css_class}">{score}/100</div>', unsafe_allow_html=True)
        st.progress(score / 100)

    # Shadow analysis
    if shadow_mode and result.get("shadow_analysis"):
        sa = result["shadow_analysis"]
        with st.expander("👻 Shadow Analysis", expanded=False):
            risk_html = _risk_badge(sa.get("hallucination_risk", "unknown"))
            st.markdown(f"**Hallucination Risk:** {risk_html}", unsafe_allow_html=True)

            if sa.get("weaknesses"):
                st.markdown("**Weaknesses**")
                for w in sa["weaknesses"]:
                    st.markdown(f"- {w}")

            if sa.get("missing_information"):
                st.markdown("**Missing Information**")
                for m in sa["missing_information"]:
                    st.markdown(f"- {m}")

            if sa.get("contradictions"):
                st.markdown("**Contradictions**")
                for c in sa["contradictions"]:
                    st.markdown(f"- {c}")

    # Fact-check report
    if fact_check_mode and result.get("fact_check"):
        with st.expander("🧾 Fact Check Report", expanded=True):
            fc = result["fact_check"]
            # Header
            header_cols = st.columns([3, 1.5, 3, 2])
            header_cols[0].markdown("**Claim**")
            header_cols[1].markdown("**Status**")
            header_cols[2].markdown("**Justification**")
            header_cols[3].markdown("**Evidence**")
            st.markdown("---")
            for item in fc:
                row = st.columns([3, 1.5, 3, 2])
                row[0].markdown(item["claim"])
                row[1].markdown(_status_badge(item["status"]), unsafe_allow_html=True)
                row[2].markdown(item.get("justification", "—"))
                evidence_text = item.get("evidence", "—")
                row[3].markdown(
                    f'<small>{evidence_text[:200]}{"…" if len(evidence_text) > 200 else ""}</small>',
                    unsafe_allow_html=True,
                )

    # Sources
    if result.get("sources"):
        with st.expander("📚 Sources", expanded=False):
            for src in result["sources"]:
                st.markdown(f"- 📄 `{src}`")

    # Raw JSON download
    st.download_button(
        label="⬇ Download JSON",
        data=json.dumps(result, indent=2),
        file_name="shadow_rag_result.json",
        mime="application/json",
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(store: EmbeddingStore):
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        st.markdown("### 📂 Documents")
        uploaded = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Files are saved to /data and indexed automatically.",
        )
        if uploaded and st.button("📥 Ingest Files"):
            with st.spinner("Ingesting and indexing…"):
                for f in uploaded:
                    save_uploaded_file(f)
                names = build_index_from_data(store)
                st.session_state.index_built = store.is_ready
                st.session_state.doc_names = names
                st.cache_resource.clear()
            st.success(f"Indexed {len(uploaded)} file(s)!")

        if st.button("🔄 Re-index /data folder"):
            with st.spinner("Re-indexing…"):
                names = build_index_from_data(store)
                st.session_state.doc_names = names
                st.cache_resource.clear()
            st.success("Index rebuilt!")

        status_color = "🟢" if store.is_ready else "🔴"
        chunk_count = len(store.chunks) if store.is_ready else 0
        st.markdown(f"{status_color} Index: **{chunk_count} chunks**")

        st.markdown("---")
        st.markdown("### 🔬 Analysis Modes")
        shadow_mode = st.toggle("👻 Shadow Mode", value=True, help="Run self-critical analysis.")
        fact_check_mode = st.toggle("🧾 Fact Check", value=True, help="Verify each claim independently.")

        st.markdown("---")
        st.markdown("### 🎛️ Retrieval Settings")
        top_k = st.slider("Top-K chunks", min_value=2, max_value=10, value=5)

        st.markdown("---")
        st.markdown("### 🤖 LLM Backend")
        backend = os.getenv("LLM_BACKEND", "gemini")
        st.markdown(f"**Active:** `{backend}`")
        st.caption("Change via `LLM_BACKEND` env var.")

    return shadow_mode, fact_check_mode, top_k


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    _init_state()

    st.markdown('<div class="main-title">👻 Shadow RAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Self-Critical Retrieval · Claim-Level Verification · Confidence Scoring</div>',
        unsafe_allow_html=True,
    )

    store = get_embedding_store()
    llm = get_llm()

    shadow_mode, fact_check_mode, top_k = render_sidebar(store)

    # Show warning if backend not configured
    if llm is None:
        st.warning(
            "⚠️ LLM not initialised. Set `GEMINI_API_KEY` in your environment, "
            "then refresh the page. Or set `LLM_BACKEND=huggingface`."
        )
        return

    if not store.is_ready:
        st.info(
            "📂 No documents indexed yet. Upload files in the sidebar or place them in the `data/` "
            "folder and click **Re-index**."
        )
        return

    retriever = Retriever(store)

    # Question input
    question = st.text_area(
        "❓ Ask a question about your documents",
        height=100,
        placeholder="What are the main findings of the report?",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=False):
            st.session_state.last_result = None
            st.rerun()

    if ask_btn and question.strip():
        with st.spinner("Thinking… this may take 20–60 seconds."):
            start = time.time()
            try:
                result = run_pipeline(
                    question=question.strip(),
                    retriever=retriever,
                    llm=llm,
                    shadow_mode=shadow_mode,
                    fact_check_mode=fact_check_mode,
                    top_k=top_k,
                )
                elapsed = time.time() - start
                st.session_state.last_result = result
                st.caption(f"Completed in {elapsed:.1f}s")
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                logger.exception("Pipeline failed")

    if st.session_state.last_result:
        render_result(st.session_state.last_result, shadow_mode, fact_check_mode)


if __name__ == "__main__":
    main()
