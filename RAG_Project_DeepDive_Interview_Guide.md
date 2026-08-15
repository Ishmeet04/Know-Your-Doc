# PDF & Excel RAG Assistant — Complete Deep-Dive & Interview Guide

A file-by-file, concept-by-concept breakdown of this project: what it is, how every piece works, how it evolved, what's broken, and how to talk about it (or rebuild it) with confidence.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [The story](#2-the-story)
3. [Tech stack at a glance](#3-tech-stack-at-a-glance)
4. [Real project structure](#4-real-project-structure)
5. [Architecture](#5-architecture)
6. [The three pipelines, in full detail](#6-the-three-pipelines-in-full-detail)
7. [File-by-file reference](#7-file-by-file-reference)
8. [RAG — everything you need to know](#8-rag--everything-you-need-to-know)
9. [LangChain — everything you need to know](#9-langchain--everything-you-need-to-know)
10. [Vector search: the actual "searching algorithm"](#10-vector-search-the-actual-searching-algorithm)
11. [Evolution & archaeology](#11-evolution--archaeology)
12. [Known issues and rough edges](#12-known-issues-and-rough-edges)
13. [Interview prep: questions and model answers](#13-interview-prep-questions-and-model-answers)
14. [Rebuilding it yourself: roadmap + requirements.txt](#14-rebuilding-it-yourself-roadmap--requirementstxt)
15. [Quick glossary](#15-quick-glossary)

---

## 1. Executive summary

This is a small **"chat with your documents" backend**: a Flask API with three endpoints that let a user (a) summarize a PDF, (b) have a multi-turn conversation with a PDF using Retrieval-Augmented Generation (RAG), and (c) ask natural-language questions of an Excel spreadsheet. A Streamlit app is the frontend. Google's Gemini models do the language understanding; LangChain orchestrates the PDF pipeline; ChromaDB stores vector embeddings for retrieval.

The interesting thing for an interview is that **the codebase actually contains two different architectural patterns living side by side**: the PDF path is "real" RAG (chunk → embed → store → retrieve → generate), while the Excel path skips RAG entirely and hands the whole file to Gemini's native file understanding. Knowing *why* those are different — and being able to argue both are legitimate choices for their respective data types — is the single highest-value thing you can walk into an interview knowing.

## 2. The story

Nothing in the repo states the original author's intent directly, so here's what the code itself tells you, and it tells a fairly clear story. The conversational RAG chain in `qa_RAG.py` is built from `create_history_aware_retriever` + `create_stuff_documents_chain` + `create_retrieval_chain` + `RunnableWithMessageHistory` — that exact combination, with those exact function names and that exact "contextualize the question, then answer from context" two-prompt structure, is the pattern in LangChain's own official Conversational RAG documentation. That's not a coincidence; it's a strong signal this project started life as someone learning RAG by working through LangChain's tutorials and then wrapping the result in a small Flask + Streamlit app to make it usable — an extremely common and legitimate way real engineers pick up a new framework.

That framing matters for how you talk about this project: it's not a novel architecture you invented, it's a **solid, correctly-assembled implementation of a well-known pattern**, extended with a second, different pattern (the Excel path) that isn't just copied from a tutorial. Both of those are genuinely good interview material — one shows you can learn and correctly apply a framework's idioms, the other shows independent judgment about when *not* to use the "obvious" tool.

## 3. Tech stack at a glance

| Layer | Technology | Role |
|---|---|---|
| Frontend | Streamlit | File upload UI, chat display, calls Flask over HTTP |
| Backend API | Flask | Three REST endpoints, request handling |
| Orchestration | LangChain (`langchain-classic`, `langchain-core`, `langchain-community`, `langchain-text-splitters`) | Prompt templates, chains, retrievers, chat memory |
| LLM + embeddings (LangChain path) | `langchain-google-genai` → Gemini | Chat completions and text embeddings for the PDF path |
| LLM (raw SDK path) | `google.generativeai` → Gemini | Native file understanding for the Excel path |
| Vector database | ChromaDB (via `langchain-chroma`) | Stores chunk embeddings, does similarity search |
| PDF parsing | PyPDF2 | Extracts raw text from PDF pages |
| Excel handling | pandas | Reads `.xlsx`, converts to CSV |
| Config | python-dotenv | Loads `GOOGLE_API_KEY` from `.env` |
| Session state (backend) | Plain Python `dict`s | In-memory storage of vectorstores and chat history |
| Session state (frontend) | `st.session_state` | Per-browser UUID, chat display history |

## 4. Real project structure

The files you were given are flat, but the import statements reveal the actual package layout the code expects (everything imports as `backend.<subpackage>.<module>`, so `backend/` must be a top-level package sitting next to `app.py`):

```
project-root/
├── app.py                          # Flask entrypoint
├── streamlit_app.py                # Frontend (standalone, calls Flask via HTTP)
└── backend/
    ├── __init__.py
    ├── services/                   # Orchestrates a full use-case
    │   ├── summary_service.py
    │   ├── q_a.py
    │   └── csv_qa.py
    ├── database/                   # State: vector store + session dict
    │   ├── chroma_setup.py
    │   ├── session_db.py
    │   └── list_collections.py     # standalone debug script
    ├── models/                     # LLM chain construction
    │   ├── qa_RAG.py
    │   ├── summarization_chain.py
    │   └── csv_handle.py
    └── PreProcessors/               # Raw file → usable data
        ├── pdf_to_text.py
        ├── textToChunks.py
        └── xl2csv.py
```

Worth noticing for the interview: this is a fairly clean separation of concerns for a learning project — **PreProcessors** turn raw files into text/documents, **models** wrap LLM calls into chains, **database** holds state, and **services** compose the other three layers into one callable function per use case. That's a legitimate, defensible layering even though nothing enforces it (there's no dependency-injection framework, no interfaces — it's just consistent convention).

## 5. Architecture

Three tiers, with a fork in the last one:

```
Browser
   │
   ▼
Streamlit server (Python)  ── st.session_state: page, session_id (UUID), chat_history
   │  HTTP POST (multipart form: file + question/query + session_id)
   ▼
Flask API  ── /summarize   /qa   /process_excel
   │                           │
   ├── PDF path ───────────────┤
   │   PDFToText → TextToChunks → Chroma (embed + store) → retriever
   │   → history-aware retriever → stuff-documents chain → Gemini (via LangChain)
   │
   └── Excel path
       pandas (xlsx → csv) → upload file to Gemini's Files API
       → Gemini reasons over the file directly (no chunking, no vector store)
```

**Why no browser-side CORS problem, even though Streamlit and Flask run on different ports:** Streamlit's `requests.post(...)` calls happen inside the Streamlit *server* process, not in JavaScript running in the user's browser. The browser only ever talks to Streamlit. CORS is a browser-enforced restriction on cross-origin *client-side* requests, so it simply doesn't apply to server-to-server calls. This trips people up constantly — good to have a crisp answer ready.

**The one thing to really understand:** the PDF path and the Excel path are both "let an LLM answer questions about a file," but they solve the "how does the model see the file" problem in opposite ways. PDF Q&A pre-processes the document into small labeled pieces so only the *relevant* pieces get sent to the model per question. Excel Q&A sends the *entire* file to the model every time and lets Gemini's native multimodal file understanding parse it directly. Neither is objectively "better" — see [section 8](#8-rag--everything-you-need-to-know) for the trade-offs.

## 6. The three pipelines, in full detail

### 6.1 PDF summarization — `POST /summarize`

1. Streamlit uploads the PDF as multipart form data.
2. `app.py` reads the raw bytes into a `BytesIO` object (no disk write).
3. `summary_service.summarize_pdf(file_content)`:
   - `PDFToText(file_content).extract_text()` — PyPDF2 opens the in-memory stream, loops every page, concatenates `page.extract_text()` for each.
   - `get_summarization_chain()` builds an LCEL pipeline: `PromptTemplate | ChatGoogleGenerativeAI | StrOutputParser`. The prompt is a single "Write a concise summary of the following: {text} CONCISE SUMMARY:" template.
   - `.invoke(extracted_text)` runs the whole extracted text through the model in **one** call.
4. Flask returns `{message, summary}` as JSON.

**No chunking, no retrieval — this is "stuff the whole document into one prompt."** That's fine as long as the PDF's text fits comfortably inside the model's context window; for a very long document it would need to become a map-reduce or refine-style summarization chain (LangChain has both built in) instead of a single "stuff" call.

**One thing worth testing, not assuming:** the chain is invoked with a bare string (`chain.invoke(extracted_text)`) rather than a dict (`chain.invoke({"text": extracted_text})`), even though the prompt template's only variable is named `text`. Whether that "just works" depends on the exact LangChain version's convenience handling for single-variable prompts. If you rebuild this and hit a `KeyError` or validation error here, that's the first place to look — the fix is simply passing `{"text": extracted_text}`.

### 6.2 PDF conversational Q&A — `POST /qa` (the real RAG path)

This is the most sophisticated part of the codebase and the one worth being able to draw on a whiteboard from memory.

1. Flask reads the file into `BytesIO`, pulls `question` and `session_id` from the form.
2. `q_a(file_content, question, session_id)` checks `session_store` (a plain dict in `session_db.py`) for this `session_id`:
   - **First question in a session:** extract text (`PDFToText`) → split into chunks (`TextToChunks`, using LangChain's `CharacterTextSplitter`: split on `\n`, 2,500-character chunks, 200-character overlap) → embed and store each chunk in a **new** Chroma collection named `langchain_{session_id}` → cache `{'vectorstore': vectorstore}` in `session_store[session_id]`.
   - **Every question after that:** skip re-processing entirely and reuse the cached vectorstore. (The frontend still re-uploads the same file bytes on every question since Streamlit keeps them in memory, but the backend just ignores the file on cache hits — a little wasteful over the wire, harmless in effect.)
3. `retriever = vectorstore.as_retriever()` — wraps Chroma's similarity search in LangChain's standard `Runnable` retriever interface.
4. `create_rag_chain(retriever)` builds a two-stage chain:
   - **Stage A — history-aware retriever.** Given the chat history and the new question, one LLM call rewrites the question into a standalone version that doesn't depend on prior turns (e.g. "how much does it cost?" → "how much does the annual subscription cost?"). That rewritten question — not the raw one — is what actually gets embedded and searched against Chroma. This is the piece that makes follow-up questions work at all; without it, "what about its cost?" would be searched for almost literally and retrieve garbage.
   - **Stage B — stuff-documents chain.** The retrieved chunks (as `{context}`), the chat history, and the *original* question go into a second prompt instructing the model to answer only from the given context, in three sentences or fewer, and to say "I don't know" if the answer isn't there. This is the actual answer-generation step.
5. `invoke_rag_chain` wraps the chain in `RunnableWithMessageHistory`, which automatically injects and updates conversation history keyed by `session_id`, backed by a **second, separate** in-memory dict (`store` inside `qa_RAG.py` — not the same dict as `session_store` in `session_db.py`, even though both are keyed by the same `session_id`). It returns `result["answer"]`.

Two LLM calls happen per question: one to rewrite the question, one to answer it. That's the price of making multi-turn conversation work over a vector store.

### 6.3 Excel/CSV Q&A — `POST /process_excel` (the non-RAG path)

1. Flask **saves the uploaded file to disk** this time (`database/files/<filename>`) — unlike the PDF path, which never touches disk.
2. `process_excel_for_qa(file_path, query, session_id)`:
   - `excel_to_csv()` — pandas reads the Excel file and writes it out as `database/files/<session_id>.csv`.
   - `handle_csv_query(csv_path, query)` — uses the **raw** `google.generativeai` SDK (not LangChain): uploads the CSV to Gemini's Files API, calls `model.generate_content([uploaded_file, query])` so Gemini reads and reasons over the file natively, then deletes the file from Google's servers.
   - Both the temporary CSV and the original uploaded Excel file are deleted from local disk afterward.
3. Flask returns `{message: response_text}`.

**No chunking. No embeddings. No vector store.** The entire spreadsheet goes to the model every single call. This is a completely different, equally valid strategy: tabular data has structure (rows, columns, headers) that arbitrary text-chunking can silently destroy — split a table in the wrong place and you separate a row from its header, or a value from the row it belongs to. For data small enough to fit in context, "just give the model the whole file" sidesteps that failure mode entirely and is simpler to build and debug. RAG earns its complexity when the corpus is too big for context or when you need low-latency retrieval over a large, largely-static knowledge base — a single uploaded spreadsheet is neither.

## 7. File-by-file reference

| File | Purpose | Worth knowing |
|---|---|---|
| `app.py` | Flask entrypoint, 3 routes | Loads `.env` from the parent directory; creates `database/files/` on startup; `app.run(debug=True)` with no port set → defaults to **5000** |
| `chroma_setup.py` | Builds a Chroma vectorstore per session | `chroma_client = Chroma()` at module level is dead code — `setup_chroma_vector_db` creates its own store via `Chroma.from_documents()` instead. No `persist_directory` is set, so storage is in-memory only |
| `session_db.py` | Vectorstore cache | A plain `dict`, `{session_id: {'vectorstore': ...}}` |
| `csv_handle.py` | Excel Q&A via raw Gemini SDK | Uses `google.generativeai`, which is fully deprecated (see [§12](#12-known-issues-and-rough-edges)) |
| `qa_RAG.py` | Conversational RAG chain construction | The history-aware retriever + stuff-documents chain pattern; owns its own `store` dict for chat history, separate from `session_db.py`'s |
| `summarization_chain.py` | "Stuff" summarization chain | Single `PromptTemplate \| llm \| StrOutputParser` pipeline |
| `textToChunks.py` | Wraps `CharacterTextSplitter` | 2,500-char chunks, 200-char overlap, split on `\n`; the currently-active version doesn't attach any per-chunk metadata (the commented-out earlier version did) |
| `pdf_to_text.py` | PyPDF2-based text extraction | Concatenates page text with **no separator** — text from the end of one page can run into the start of the next with no space |
| `xl2csv.py` | Excel → CSV via pandas | Thin wrapper around `read_excel` / `to_csv` |
| `summary_service.py` | Orchestrates PDF summarization | Passes a `BytesIO` stream straight through — no disk write |
| `csv_qa.py` | Orchestrates Excel Q&A | Deletes both the temp CSV and the original upload after answering; builds its path with a hardcoded forward slash (`"database/files"`) rather than `os.path.join`, unlike `app.py` |
| `q_a.py` | Orchestrates PDF conversational Q&A | The cache-check-then-build-or-reuse logic for the vectorstore |
| `list_collections.py` | Standalone debug script | Not part of the app's request flow; contains a call (`chroma_client._collection.get("fsfwd")`) that looks like leftover experimentation, not working code |
| `streamlit_app.py` | Frontend | Contains its own entire earlier version, commented out at the top — see [§11](#11-evolution--archaeology) |
| `__init__.py` × 3 | Package markers | Empty; make the `backend` subpackages importable |

## 8. RAG — everything you need to know

**The problem RAG solves.** LLMs are frozen at training time and have a limited context window. RAG lets a model answer questions about documents it never trained on and that may be far larger than its context window, without any retraining — by fetching only the relevant pieces of a document at query time and inserting them into the prompt.

**The pipeline:**
1. **Ingest** — get the raw content (here: extract PDF text).
2. **Chunk** — split it into smaller pieces. Why not embed the whole document as one vector? Because a single embedding for a 50-page PDF would be a blurry average of everything in it, matching almost nothing precisely. Smaller chunks give more precise, targeted retrieval.
3. **Embed** — convert each chunk into a numeric vector that captures its meaning (see §10).
4. **Store** — put those vectors in a vector database that can search by similarity, not just keyword.
5. **Retrieve** — at query time, embed the *question* with the same embedding model, then find the chunks whose vectors are closest to it.
6. **Augment** — insert the retrieved chunks into the prompt as context.
7. **Generate** — the LLM answers, grounded in that context instead of (or in addition to) its own training data.

**Why chunk overlap matters.** If a sentence that answers the question straddles the boundary between chunk 4 and chunk 5, splitting with zero overlap can leave neither chunk with the complete idea. A 200-character overlap (as used here) means the last ~200 characters of one chunk repeat as the first ~200 of the next, so boundary-straddling content survives in at least one full chunk.

**Why the history-aware retriever exists.** Naive RAG searches with the user's raw question. That breaks the moment a conversation has follow-ups: "What's the termination clause?" then "How long is the notice period?" is fine standalone, but "What about for the other party?" means nothing to a similarity search on its own — it has to be rewritten to something like "What is the notice period for the other party's termination?" before it's searched. This project's history-aware retriever does exactly that rewrite with a dedicated LLM call before ever touching the vector store. This is the single most-copied production RAG pattern for a reason — it's cheap (one extra small LLM call) and fixes a failure mode that would otherwise make every multi-turn RAG chat feel broken by the second question.

**Common RAG failure modes** (great material for "what would you improve" questions):
- **Irrelevant retrieval** — if chunk size or embedding model quality is poor, the "most similar" chunks aren't actually the most *useful* ones.
- **Lost in the middle** — LLMs pay less attention to information buried in the middle of a long context, so stuffing in too many retrieved chunks can hurt as much as it helps.
- **Confident hallucination despite context** — the model can still ignore or misread the retrieved text and answer from its own (possibly wrong) training knowledge.
- **Stale index** — if the source document changes, the vector store doesn't know unless you re-ingest.
- **Chunk-boundary damage** — especially bad for structured data (tables, code, spreadsheets), which is exactly why this project's Excel path avoids chunking altogether.

**Beyond basic top-k retrieval** (good "what else is out there" talking points): re-ranking retrieved chunks with a separate cross-encoder model, hybrid search (combining keyword/BM25 search with vector search so exact terms like product codes aren't lost), Maximal Marginal Relevance (MMR — retrieves for relevance *and* diversity, avoiding five near-duplicate chunks), parent-document retrievers (search small chunks but return their larger surrounding context), self-querying retrievers (the LLM extracts structured filters from the question), and evaluation frameworks like RAGAS that score faithfulness (is the answer actually supported by the retrieved context) and retrieval precision/recall separately from generation quality.

## 9. LangChain — everything you need to know

**What it is.** A framework for composing LLM calls with prompts, external data, memory, and tools into repeatable pipelines, so you're not hand-writing string concatenation and API calls for every step.

**LCEL (LangChain Expression Language).** The `|` pipe operator composes `Runnable` objects into a pipeline: `prompt | llm | StrOutputParser()` means "format the prompt, send it to the model, parse the raw response into plain text." Every piece implements a common interface (`.invoke()`, `.stream()`, `.batch()`), which is what lets you snap them together like this.

**Building blocks used in this project:**
- `PromptTemplate` / `ChatPromptTemplate` — templated prompts with `{variable}` placeholders.
- `MessagesPlaceholder("chat_history")` — a slot inside a prompt that gets filled with a whole list of prior messages, not a single string.
- `StrOutputParser` — extracts the plain text string out of the LLM's structured response object.
- **Chain factories**: `create_history_aware_retriever`, `create_stuff_documents_chain`, `create_retrieval_chain` — pre-built functions that wire the prompt + LLM + retriever pieces above into a working chain, so you don't hand-assemble them.
- **Retrievers** — `vectorstore.as_retriever()` adapts a vector database into the standard "text in, relevant documents out" `Runnable` interface every chain expects.
- `RunnableWithMessageHistory` + `BaseChatMessageHistory` (`ChatMessageHistory` here) — the mechanism that gives a chain "memory": every invocation is automatically wrapped with a lookup/save of that session's message history, keyed by whatever `session_id` you pass in `config`.

**Why the imports look unusual, and what that tells you.** This project imports `create_history_aware_retriever`, `create_retrieval_chain`, and `create_stuff_documents_chain` from `langchain_classic.chains` rather than `langchain.chains`. That's a real, current package — as of now, LangChain's main package has repositioned itself around building agents (with a prebuilt agent architecture, and LangGraph recommended for more custom or advanced orchestration needs), while `langchain-classic` is where the pre-agent "chain" abstractions — including exactly the ones this project uses — now live, alongside legacy `langchain-community` re-exports and other now-secondary functionality. In other words: the RAG pattern in this codebase is explicitly labeled "classic" by LangChain itself these days. That's not a criticism — it's still maintained and it's still the most direct way to build exactly this kind of pipeline — but if you rebuild this, you have a real fork in the road: keep the classic chain-based approach (matches this code, simplest, well-documented) or rewrite the orchestration in LangGraph (more current idiom, more flexible for anything beyond straight-line RAG, more code).

**Why LangChain isn't used everywhere.** The Excel path (`csv_handle.py`) calls Google's SDK directly with no LangChain involved at all. That's a legitimate choice, not an oversight: LangChain earns its keep specifically where you need prompt templating, retrieval, and memory orchestrated together — for "upload one file, ask one question, get one native-file-understanding answer," a raw SDK call is simpler and has one less layer to debug.

## 10. Vector search: the actual "searching algorithm"

This is worth being precise about, since "searching algo" is exactly the kind of thing an interviewer will probe.

**Step 1 — Embeddings.** Every chunk of text is converted into a vector (a long list of numbers, effectively coordinates in a high-dimensional space) by an embedding model. This project uses `gemini-embedding-2`. Text with similar *meaning* ends up with vectors that are close together in that space, even if the wording is completely different — "cost" and "price" land near each other; "cost" and "cat" don't. That's what makes this "search by meaning" instead of exact keyword matching.

**Step 2 — Similarity, not equality.** At query time, the question is embedded with the *same* embedding model (this matters — mixing embedding models breaks comparability, since different models don't share a coordinate space). "Closeness" is then measured with a distance metric, most commonly **cosine similarity** (the angle between two vectors — ignores magnitude, just asks "do they point the same direction") or Euclidean/L2 distance.

**Step 3 — Finding the nearest vectors efficiently.** Comparing a query vector against every single stored vector one by one (brute force) is exact but slow at scale. Production vector databases, Chroma included, use **Approximate Nearest Neighbor (ANN)** indexing — commonly a graph-based structure like **HNSW (Hierarchical Navigable Small World)** — to find *very close to* the nearest vectors in a fraction of the time, by walking a graph toward the query point instead of scanning everything.

**The honest caveat for this specific project:** the scale here doesn't actually need any of that sophistication. Each Chroma collection holds the chunks of exactly one uploaded PDF for one session — likely dozens to a few hundred chunks. At that size, brute-force comparison would be imperceptibly different in speed from ANN search. Chroma's indexing machinery is the right *default* tool because it comes with a clean API and LangChain integration out of the box, not because this workload demands ANN's scalability. That distinction — "the right tool for the job vs. the tool that happens to also work at scale" — is a good thing to be able to articulate.

**`vectorstore.as_retriever()` in practice:** when invoked with a query string, it embeds the query, runs similarity search against the collection, and returns the top-k most similar chunks (k defaults to 4 unless configured otherwise) as LangChain `Document` objects, ready to be dropped into a prompt.

## 11. Evolution & archaeology

Several files still contain their *previous* version, commented out above the active code. That's an accidental but genuinely useful changelog — here's what it shows, in the order it likely happened:

- **PDF library swap:** `pdf_to_text.py`'s commented-out version used PyMuPDF (`fitz`); the active version uses PyPDF2. (A plausible reason, though the code itself doesn't say why: PyMuPDF is AGPL-licensed, which is a common reason teams move away from it for anything that might ship commercially — but that's a reasonable guess, not something the repo confirms.)
- **Disk I/O removed from summarization:** the commented-out `summary_service.py` took a file *path* and presumably read it from disk; the active version takes a `BytesIO` stream straight from the upload, skipping a disk write/cleanup step entirely. `app.py`'s `/summarize` route matches the newer, streaming-friendly version.
- **LangChain's own internal restructuring shows up in the diffs:** `textToChunks.py`'s commented-out version imports `Document` from `langchain.docstore.document`; the active version imports it from `langchain_core.documents` — tracking LangChain's real, well-documented split of core abstractions out of the monolithic `langchain` package into `langchain-core`.
- **Streamlit API migration:** every `st.experimental_rerun()` in the old commented block became `st.rerun()` in the active code — Streamlit stabilized and renamed that API a while back.
- **The Excel workflow gained a real query parameter:** the old commented Streamlit code posted to `/process_excel` with just a file, no query and no session_id. The active version adds a `query` text input and only enables the "Process Excel" button once both a file *and* a query are present, and sends both `query` and `session_id` in the request. That's the difference between "process this file somehow" and an actual targeted Q&A feature — a real capability upgrade, not just a refactor.
- **A bug the evolution left behind:** the old Streamlit code posted to `http://localhost:5000/...`; the active code posts to `http://localhost:5001/...`. But `app.py` still calls `app.run(debug=True)` with no port argument, which defaults to Flask's port **5000**. Unless you either add `port=5001` to `app.run()` or change the Streamlit URLs back to 5000, the frontend will fail to connect out of the box. This is a genuinely useful thing to know before you try to run the fork — and a great, concrete "I found and fixed a bug" story for an interview.

## 12. Known issues and rough edges

Worth knowing cold — these are the questions most likely to come up, and they're all real:

- **Everything stateful is in-memory and ephemeral.** Both the vectorstore cache (`session_db.py`) and the chat-history store (inside `qa_RAG.py`) are plain Python dicts. Restart the Flask process and every session is gone. Run more than one worker process (which any real production deployment would) and requests for the same `session_id` can land on a worker that's never seen that session, silently "forgetting" everything.
- **Chroma isn't persisted to disk.** No `persist_directory` is configured, so vector data lives only in process memory, on top of already being wiped on restart.
- **Two separate, unlinked "session" dicts** track different things under the same `session_id` — one for the vectorstore, one for chat history — which works but is a subtle trap for future bugs if only one gets cleared or migrated.
- **The Excel and PDF backend port disagree** (see §11) — a real, reproducible bug in the current fork.
- **`csv_handle.py` uses a fully deprecated SDK.** `google.generativeai` reached end-of-life on November 30, 2025 — no further updates or bug fixes. It may still run today, but Google has been aggressively retiring old model endpoints (Gemini 1.0 and 1.5 are already fully shut down; 2.0 Flash followed in June 2026), so this is a ticking clock, not a stable foundation. Migrating to the new unified `google-genai` SDK is the fix (see §14).
- **`debug=True` in `app.run()`** enables Werkzeug's interactive debugger, which allows arbitrary code execution if the port is ever exposed beyond localhost — fine for local development, a real problem if deployed as-is.
- **No authentication, no file-size limits, minimal input validation.** Anyone who can reach the Flask port can call any endpoint; there's no `MAX_CONTENT_LENGTH`, and the Excel endpoint doesn't check the uploaded file's extension before saving it.
- **Thin error handling.** Most routes have no `try/except`; an exception surfaces as a raw 500 with a stack trace (fine in `debug=True` local dev, not fine otherwise).
- **`chroma_client = Chroma()`** at the top of `chroma_setup.py` is dead code — nothing reads from it; the real work happens in a separately-created store inside `setup_chroma_vector_db`.
- **`list_collections.py`** is a leftover debug script, not part of the request flow, and its one active line (`chroma_client._collection.get("fsfwd")`) looks like an abandoned experiment rather than working code.
- **A likely page-boundary text bug:** `pdf_to_text.py` concatenates page text with no separator between pages, so the last word of one page can run directly into the first word of the next with no space — unlike the commented-out PyMuPDF version, which joined pages with `"\n"`.
- **Path-building inconsistency:** `app.py` builds its upload folder path with `os.path.join('database', 'files')` (portable across OSes); `csv_qa.py` hardcodes `"database/files"` with a forward slash. They resolve the same on Linux/macOS but could diverge on Windows.

## 13. Interview prep: questions and model answers

**Be upfront that this is a fork you went deep on.** That's a stronger answer than pretending you invented every architectural decision — interviewers care much more about whether you understand the *trade-offs* than whether you typed every line first. A good opening line: *"I forked this to learn RAG hands-on, then went through every module to understand exactly how it works and where I'd change it."* Then demonstrate that depth with answers like these:

**Q: Walk me through what happens when a user uploads a PDF and asks a question.**
A: The file streams into Flask as bytes, no disk write. If it's a new session, PyPDF2 extracts the text, it's split into 2,500-character chunks with 200-character overlap, each chunk is embedded and stored in a per-session Chroma collection. The question first goes through a history-aware retriever — one LLM call rewrites it into a standalone version using chat history — then that rewritten question is embedded and matched against the stored chunks via similarity search. The top matches, the original question, and the chat history all go into a second LLM call that generates the final answer, constrained to only use the retrieved context.

**Q: Why RAG instead of just pasting the whole document into the prompt?**
A: Context window limits, cost, and precision. A large document may not fit at all; even when it does, sending the whole thing on every question is expensive and dilutes the model's attention with irrelevant text. RAG retrieves only what's likely relevant to *this specific question*. (Worth noting: the summarization endpoint in this same project actually *does* stuff the whole document in — a good example of picking the simpler approach when the task doesn't need retrieval.)

**Q: Why doesn't the Excel path use RAG?**
A: Tabular data has structure that naive text-chunking can destroy — split a table wrong and you separate a value from its row or header. The whole spreadsheet is small enough to fit in context, so the simpler, safer choice is giving Gemini the entire file through its native file-understanding API and letting it reason over the intact structure directly.

**Q: How does the app handle multi-turn conversation — how does it know what "it" refers to?**
A: Two mechanisms working together: `RunnableWithMessageHistory` automatically loads and appends each session's message history around every call, and the history-aware retriever uses that history to rewrite ambiguous follow-ups into standalone questions *before* searching the vector store.

**Q: What are the weaknesses of the session management here?**
A: It's two separate in-memory Python dicts with no persistence, no expiration, and no cross-process sharing — fine for a single-process demo, but it loses all data on restart and breaks the moment you run more than one server process, since each process has its own independent dict.

**Q: How would you scale this to production?**
A: Replace the in-memory dicts with a real store (Redis for chat history and session metadata; Chroma configured with a persistent directory, or a hosted vector DB, for embeddings). Move file processing for large uploads to a background task queue instead of blocking the request. Add auth, request size limits, and structured error handling. Run behind a real WSGI server instead of Flask's dev server.

**Q: What's the difference between how the PDF path and the Excel path call Gemini?**
A: The PDF path goes through LangChain's `ChatGoogleGenerativeAI` wrapper, which is the currently-maintained integration path. The Excel path calls the raw `google.generativeai` SDK directly — which is fully deprecated in favor of the new unified `google-genai` SDK, so that part specifically needs updating if this were going into production today.

**Q: What would you change if you rebuilt this from scratch?**
A: Persist the vector store and session state, fix the port mismatch between the Flask app and the Streamlit frontend, migrate off the deprecated Gemini SDK, use a splitter that respects semantic boundaries better than a flat character count, add real error handling and input validation, and decide deliberately whether to keep the classic LangChain chain pattern or move the orchestration to LangGraph.

**Q: What would you test?**
A: Chunking behavior on edge cases (empty PDF, single-page PDF, a PDF that's mostly images with no extractable text), the retriever with mocked embeddings so tests don't hit the real API, the two session dicts staying in sync, and the Excel endpoint against malformed or non-UTF8 spreadsheet data.

## 14. Rebuilding it yourself: roadmap + requirements.txt

Follow the six-step order above (environment → PDF extraction → summarization → chunking/vector store → conversational RAG → Excel path + hardening) rather than building everything at once — each stage is independently testable before the next depends on it.

**A starting `requirements.txt`** — verify exact current versions when you actually install, since this ecosystem moves fast:

```
flask
python-dotenv
streamlit
requests
pandas
openpyxl
pypdf
langchain-core
langchain-classic
langchain-text-splitters
langchain-google-genai
langchain-chroma
langchain-community
chromadb
google-genai
```

**Two changes that are worth making deliberately, not just copying the original:**

1. **Swap `google-generativeai` for `google-genai`** in the Excel path. The calling pattern changes shape — roughly:
   ```python
   from google import genai
   client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
   uploaded = client.files.upload(file=csv_path)
   response = client.models.generate_content(model="gemini-3.6-flash", contents=[uploaded, query])
   result = response.text
   client.files.delete(name=uploaded.name)
   ```
   Treat this as a sketch, not copy-paste-ready code — check the current `google-genai` docs for exact method signatures before relying on it.

2. **Model names to use today:** for chat, `gemini-3.6-flash` is the current recommended default (faster and cheaper than `gemini-3.5-flash`, which this project already uses and which still works fine); for embeddings, `gemini-embedding-2` (already used in this project, and genuinely current — it's Google's first multimodal embedding model) or `gemini-embedding-001` if you only need text. Both the chat and embedding model names already in this codebase turned out to be real, current models rather than placeholders — worth double-checking against `ai.google.dev/gemini-api/docs/models` at build time regardless, since this space changes monthly.

**Fixes to bake in from the start rather than patch later:**
- Configure Chroma with a `persist_directory` so embeddings survive a restart.
- Back session state (both the vectorstore cache and chat history) with Redis or a lightweight database instead of module-level dicts.
- Make the Flask port and the Streamlit target URL match (and ideally read the port from an environment variable in both places instead of hardcoding it twice).
- Use `RecursiveCharacterTextSplitter` instead of `CharacterTextSplitter` — it tries multiple separators (paragraph, then sentence, then word) in order, which tends to produce more semantically coherent chunks than a flat character-count split.
- Wrap the file-processing routes in `try/except` and return meaningful error responses instead of letting exceptions surface as raw 500s.

## 15. Quick glossary

- **RAG (Retrieval-Augmented Generation)** — answering questions by retrieving relevant source material at query time and feeding it to an LLM, instead of relying only on what the model learned during training.
- **Embedding** — a numeric vector representation of text (or other data) that captures meaning, such that semantically similar content has vectors that are close together.
- **Vector store / vector database** — a database built to store embeddings and search them by similarity instead of exact match.
- **Chunk** — a piece of a larger document, split up so retrieval can return just the relevant part instead of the whole thing.
- **Retriever** — an abstraction over "given a query, return relevant documents," regardless of what's underneath it.
- **LCEL** — LangChain Expression Language; the `|` syntax for composing chain components.
- **Chain** — a composed pipeline of prompt → model → parser (and often more) that runs as one unit.
- **Session** — server-side state tied to one user's ongoing interaction, here identified by a UUID generated client-side.
- **Top-k** — the number of most-similar results a retriever returns (default 4 in LangChain unless set otherwise).
- **Context window** — the maximum amount of text an LLM can consider at once, prompt and history included.
- **Cosine similarity** — a common way to measure how close two vectors are, based on the angle between them.
- **ANN (Approximate Nearest Neighbor)** — a family of algorithms (HNSW among them) that find *very close to* the nearest vectors quickly, trading a little precision for a lot of speed at scale.
- **Token** — the unit LLMs process text in; roughly ¾ of a word on average in English.
