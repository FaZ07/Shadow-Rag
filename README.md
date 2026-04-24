# 👻 Shadow RAG — Self-Critical Retrieval System

> A production-grade RAG system that doesn't just answer — it **doubts itself**.

---

## What Is Shadow RAG?

Standard RAG retrieves documents and generates an answer. Shadow RAG goes further:

1. **Answers** using only retrieved context (no hallucination)
2. **Critiques** its own answer via a Shadow Layer (identifies weaknesses, gaps, hallucination risk)
3. **Verifies** every atomic claim in the answer independently
4. **Scores** the overall trustworthiness (0–100)

The result is a structured, auditable AI response — not a black box.

---

## Why It's Different from Normal RAG

| Feature | Standard RAG | Shadow RAG |
|---|---|---|
| Answer generation | ✅ | ✅ |
| Source attribution | ✅ | ✅ |
| Self-critique | ❌ | ✅ |
| Claim-level verification | ❌ | ✅ |
| Hallucination risk score | ❌ | ✅ |
| Confidence scoring | ❌ | ✅ |

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│  Document Ingestion  │ ← PDF, TXT from /data
│  + Semantic Chunking │   (sentence-aware, with overlap)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Embedding Layer     │ ← sentence-transformers (all-MiniLM-L6-v2)
│  + FAISS Index       │   stored locally, reloaded on restart
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Retrieval Layer     │ ← cosine similarity, top-k chunks
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Answer Generation   │ ← LLM (Gemini / HuggingFace)
│  (context-grounded)  │   strict prompt: use ONLY context
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌──────────────────────────┐
│Shadow │  │  Claim Verification Loop  │
│Critique│  │  1. Extract atomic claims │
│Layer  │  │  2. Re-retrieve per claim │
│       │  │  3. Verify: VERIFIED /    │
│       │  │     PARTIALLY / NOT_SUPP  │
└───┬───┘  └────────────┬─────────────┘
    │                    │
    └──────────┬─────────┘
               ▼
    ┌─────────────────────┐
    │  Confidence Score    │ ← 0–100 based on claims + risk
    └─────────────────────┘
               │
               ▼
         JSON Output
```

---

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/yourname/shadow-rag
cd shadow-rag
pip install -r requirements.txt
```

### 2. Configure LLM

Copy `.env.example` → `.env` and fill in your key:

```bash
cp .env.example .env
```

**Option A — Gemini (recommended, free):**
Get a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

```env
LLM_BACKEND=gemini
GEMINI_API_KEY=your_key_here
```

**Option B — Local HuggingFace model (no API key):**

```env
LLM_BACKEND=huggingface
HF_MODEL=google/flan-t5-large
```

### 3. Add Documents

Place your `.pdf` or `.txt` files in the `data/` folder:

```
data/
├── research_paper.pdf
├── company_report.txt
└── ...
```

### 4. Run

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## UI Walkthrough

1. **Upload documents** via the sidebar file uploader, or place them in `data/` and click **Re-index**.
2. **Toggle Shadow Mode** to enable/disable self-critique.
3. **Toggle Fact Check** to enable/disable claim-level verification.
4. Type your question and click **Ask**.
5. View:
   - 🧠 Answer with source citations
   - 📊 Confidence score (0–100) with progress bar
   - 👻 Shadow Analysis (weaknesses, gaps, contradictions, hallucination risk)
   - 🧾 Fact Check table (color-coded: green/yellow/red)
   - 📚 Source documents used

---

## Example Output

```json
{
  "question": "What were the main causes of the 2008 financial crisis?",
  "answer": "According to the documents, the main causes were...",
  "confidence_score": 74,
  "shadow_analysis": {
    "weaknesses": ["The answer does not quantify the scale of subprime exposure."],
    "missing_information": ["Long-term regulatory context is absent."],
    "contradictions": [],
    "hallucination_risk": "low"
  },
  "fact_check": [
    {
      "claim": "Subprime mortgage lending increased significantly between 2003 and 2006.",
      "status": "VERIFIED",
      "justification": "Document [1] explicitly states subprime lending grew 300% in that period.",
      "evidence": "\"subprime originations grew from $332B in 2003 to $1.3T in 2006\""
    }
  ],
  "sources": ["financial_crisis_report.pdf"]
}
```

---

## Project Structure

```
shadow-rag/
├── app.py                  ← Streamlit UI + pipeline orchestration
├── rag/
│   ├── __init__.py
│   ├── ingest.py           ← Document loading + semantic chunking
│   ├── embedder.py         ← Sentence-transformer embeddings + FAISS
│   ├── retriever.py        ← Top-k retrieval
│   ├── generator.py        ← LLM abstraction + answer generation
│   ├── shadow.py           ← Shadow critique layer
│   └── verifier.py         ← Claim extraction + verification + scoring
├── data/                   ← Place your documents here
├── index/                  ← Auto-generated FAISS index (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Deployment

### Streamlit Cloud (free)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Set `GEMINI_API_KEY` as a secret
4. Deploy — done

### HuggingFace Spaces (free)
1. Create a new Space with SDK: `streamlit`
2. Add your files
3. Set `GEMINI_API_KEY` in Space secrets

---

## Feature Highlights

- **Semantic chunking**: sentence-boundary aware, with configurable overlap — avoids cutting mid-thought
- **Per-claim retrieval**: each claim is verified against freshly retrieved context, not the same pool
- **Modular LLM**: swap backends via one env var — no code changes
- **Persistent index**: FAISS index saved to disk, loaded on restart — no re-embedding needed
- **Structured JSON output**: machine-readable, downloadable from the UI
- **Confidence scoring**: transparent formula (verified ratio − hallucination penalty − contradiction penalty)
