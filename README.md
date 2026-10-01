<div align="center">

# 📄 Know Your Doc

### *AI-Powered Document Intelligence — Summarize, Query & Understand Your Documents Instantly*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Google_Gemini-AI_Engine-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![LangChain](https://img.shields.io/badge/LangChain-RAG_Framework-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F61?style=for-the-badge)](https://www.trychroma.com)

---

**Know Your Doc** is a full-stack RAG (Retrieval-Augmented Generation) application that lets you upload PDF and Excel documents, get instant AI-generated summaries, and have intelligent conversations with your documents using natural language. Built with Google Gemini, LangChain, ChromaDB, and Streamlit.

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Screenshots](#-screenshots) · [Tech Stack](#-tech-stack) · [Project Structure](#-project-structure)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📑 **Multi-PDF Upload** | Upload one or multiple PDFs at once — no file limit |
| 🧠 **Auto-Summarization** | Instant AI summaries the moment files are uploaded |
| 💬 **Conversational Q&A** | Ask follow-up questions with full chat history awareness |
| 📎 **Source Citations** | Every answer cites which document(s) it came from |
| 💡 **Smart Follow-ups** | AI suggests 3 related questions after each answer |
| 📊 **Excel Processing** | Upload Excel files and query them with natural language |
| 🔄 **Session Persistence** | Resume past conversations even after server restarts |
| 🗂️ **Session Management** | Browse, resume, or start fresh sessions from the home page |

---

## 🏗 Architecture

### High-Level System Overview

```mermaid
flowchart TB
    subgraph USER["👤 User"]
        Browser["🌐 Web Browser"]
    end

    subgraph FRONTEND["🎨 Frontend — Streamlit"]
        UI["Streamlit App<br/><i>streamlit_app.py</i>"]
        Pages["Page Router<br/><i>Main · PDF · Excel</i>"]
    end

    subgraph BACKEND["⚙️ Backend — Flask"]
        API["Flask REST API<br/><i>app.py</i>"]
        
        subgraph SERVICES["Services Layer"]
            SumSvc["Summary Service"]
            QASvc["Q&A Service"]
            CSVSvc["CSV Q&A Service"]
        end

        subgraph MODELS["Models Layer"]
            SumChain["Summarization Chain"]
            RAGChain["RAG Chain<br/><i>History-Aware Retriever</i>"]
            CSVHandle["CSV Handler"]
        end

        subgraph PREPROC["PreProcessors"]
            PDF2Text["PDF → Text<br/><i>PyPDF2</i>"]
            Text2Chunk["Text → Chunks<br/><i>CharacterTextSplitter</i>"]
            XL2CSV["Excel → CSV<br/><i>pandas</i>"]
        end
    end

    subgraph DATA["💾 Data Layer"]
        ChromaDB["ChromaDB<br/><i>Vector Store</i>"]
        SQLite["SQLite<br/><i>Session + Chat History</i>"]
        InMem["In-Memory<br/><i>Session Cache</i>"]
    end

    subgraph AI["🤖 AI Services"]
        Gemini["Google Gemini 3.5 Flash<br/><i>LLM</i>"]
        Embeddings["Gemini Embedding 2<br/><i>Vector Embeddings</i>"]
    end

    Browser <--> UI
    UI --> Pages
    Pages <-->|HTTP REST| API
    
    API --> SumSvc
    API --> QASvc
    API --> CSVSvc
    
    SumSvc --> SumChain
    QASvc --> RAGChain
    CSVSvc --> CSVHandle
    
    SumSvc --> PDF2Text
    QASvc --> PDF2Text
    QASvc --> Text2Chunk
    CSVSvc --> XL2CSV
    
    RAGChain --> ChromaDB
    RAGChain --> SQLite
    QASvc --> InMem
    QASvc --> SQLite
    
    SumChain --> Gemini
    RAGChain --> Gemini
    CSVHandle --> Gemini
    Text2Chunk --> Embeddings
    ChromaDB --> Embeddings

    style USER fill:#E3F2FD,stroke:#1565C0,color:#000
    style FRONTEND fill:#FFF3E0,stroke:#E65100,color:#000
    style BACKEND fill:#E8F5E9,stroke:#2E7D32,color:#000
    style DATA fill:#F3E5F5,stroke:#6A1B9A,color:#000
    style AI fill:#FCE4EC,stroke:#C62828,color:#000
```

### RAG Pipeline — Step by Step

This is the core intelligence of the app. When you ask a question about your documents, here's exactly what happens:

```mermaid
flowchart LR
    subgraph INGEST["📥 Document Ingestion"]
        direction TB
        A["Upload PDF(s)"] --> B["Extract Text<br/><i>PyPDF2</i>"]
        B --> C["Split into Chunks<br/><i>2500 chars, 200 overlap</i>"]
        C --> D["Generate Embeddings<br/><i>Gemini Embedding 2</i>"]
        D --> E["Store in ChromaDB<br/><i>Per-session collection</i>"]
    end

    subgraph QUERY["❓ Query Pipeline"]
        direction TB
        F["User Asks Question"] --> G["Reformulate with History<br/><i>History-Aware Retriever</i>"]
        G --> H["Semantic Search<br/><i>Top-5 relevant chunks</i>"]
        H --> I["Build Context Prompt"]
        I --> J["Gemini Generates Answer"]
        J --> K["Extract Source Citations"]
        K --> L["Generate Follow-up<br/>Suggestions"]
    end

    INGEST --> QUERY

    style INGEST fill:#E8F5E9,stroke:#2E7D32,color:#000
    style QUERY fill:#E3F2FD,stroke:#1565C0,color:#000
```

### Session & Persistence Architecture

```mermaid
flowchart TD
    subgraph SESSION["🔄 Session Lifecycle"]
        New["New Session<br/><i>UUID generated</i>"] --> Upload["Files Uploaded"]
        Upload --> Embed["Vectors Created<br/><i>ChromaDB collection</i>"]
        Embed --> QA["Q&A Conversation"]
        QA --> |"Each message"| Persist["Persist to SQLite"]
        
        Resume["Resume Session"] --> LoadDB["Load from SQLite<br/><i>Chat history</i>"]
        LoadDB --> ReEmbed["Re-upload Files<br/><i>Reload vectors</i>"]
        ReEmbed --> QA
    end

    subgraph STORAGE["💾 Dual Storage Strategy"]
        InMem2["🧠 In-Memory Cache<br/><i>Fast vectorstore lookup<br/>Lost on restart</i>"]
        Chroma2["📀 ChromaDB on Disk<br/><i>Persistent vectors<br/>Survives restart</i>"]
        SQL2["🗃️ SQLite<br/><i>Sessions + Messages<br/>Permanent record</i>"]
    end

    QA --> InMem2
    Embed --> Chroma2
    Persist --> SQL2

    style SESSION fill:#FFF8E1,stroke:#F57F17,color:#000
    style STORAGE fill:#F3E5F5,stroke:#6A1B9A,color:#000
```

---

## 📸 Screenshots

<details>
<summary><b>🏠 Home Page — Choose your workflow</b></summary>
<br/>
<img src="assets/01_home_page.png" alt="Home Page" width="700"/>

> The landing page lets you choose between **PDF** and **Excel** workflows, and shows past sessions you can resume.
</details>

<details>
<summary><b>📑 PDF Upload & Auto-Summary</b></summary>
<br/>
<img src="assets/03_pdf_upload.png" alt="PDF Upload" width="700"/>
<br/><br/>
<img src="assets/04_pdf_summary.png" alt="PDF Summary" width="700"/>

> Upload one or multiple PDFs — the app **automatically summarizes** each document the moment it's uploaded.
</details>

<details>
<summary><b>💬 Q&A Conversation with Citations</b></summary>
<br/>
<img src="assets/06_qa_response.png" alt="Q&A Response" width="700"/>
<br/><br/>
<img src="assets/07_followup_suggestions.png" alt="Follow-up Suggestions" width="700"/>

> Ask questions across all your uploaded documents. Each answer comes with **source citations** and **AI-generated follow-up suggestions**.
</details>

<details>
<summary><b>📊 Excel Processing</b></summary>
<br/>
<img src="assets/11_excel_upload.png" alt="Excel Upload" width="700"/>
<br/><br/>
<img src="assets/12_excel_result.png" alt="Excel Result" width="700"/>

> Upload an Excel file and ask questions about the data in natural language.
</details>

<details>
<summary><b>🔄 Session Resume</b></summary>
<br/>
<img src="assets/10_session_resume.png" alt="Session Resume" width="700"/>

> All conversations are persisted. Come back anytime and resume where you left off.
</details>

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Google API Key** — Get one from [Google AI Studio](https://aistudio.google.com/apikey)

### 1. Clone the Repository

```bash
git clone https://github.com/Ishmeet04/Know-Your-Doc.git
cd Know-Your-Doc
```

### 2. Create & Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Start the Backend (Flask API)

```bash
python -m backend.app
```

The Flask server starts on **`http://localhost:5001`**

### 6. Start the Frontend (Streamlit UI)

In a new terminal:

```bash
streamlit run frontend/streamlit_app.py
```

The Streamlit app opens at **`http://localhost:8501`**

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Interactive web UI with chat interface |
| **Backend** | Flask | REST API server |
| **LLM** | Google Gemini 3.5 Flash | Text generation, summarization, Q&A |
| **Embeddings** | Gemini Embedding 2 | Semantic vector embeddings |
| **RAG Framework** | LangChain | Chain orchestration, history-aware retrieval |
| **Vector Store** | ChromaDB | Persistent semantic search |
| **Database** | SQLite | Session management, chat history persistence |
| **PDF Parsing** | PyPDF2 | Text extraction from PDF documents |
| **Excel Parsing** | pandas + openpyxl | Spreadsheet reading and conversion |

---

## 📁 Project Structure

```
Know-Your-Doc/
│
├── 📄 .env                          # API keys (not committed)
├── 📄 .gitignore                    # Git exclusions
├── 📄 README.md                     # You are here!
├── 📄 requirements.txt             # Python dependencies
│
├── 🎨 frontend/
│   └── streamlit_app.py             # Streamlit UI — pages, chat, file upload
│
├── ⚙️ backend/
│   ├── app.py                       # Flask REST API — routes & entry point
│   │
│   ├── 🔧 PreProcessors/           # Document pre-processing
│   │   ├── pdf_to_text.py           #   PDF → raw text (PyPDF2)
│   │   ├── textToChunks.py          #   Text → LangChain Document chunks
│   │   └── xl2csv.py                #   Excel → CSV conversion (pandas)
│   │
│   ├── 🧠 models/                   # AI model chains
│   │   ├── qa_RAG.py                #   RAG chain + follow-up suggestion gen
│   │   ├── summarization_chain.py   #   Summarization prompt chain
│   │   └── csv_handle.py            #   Direct Gemini queries on CSV files
│   │
│   ├── 📡 services/                 # Business logic layer
│   │   ├── q_a.py                   #   Full Q&A orchestration
│   │   ├── summary_service.py       #   PDF summarization service
│   │   └── csv_qa.py                #   Excel → CSV → Gemini pipeline
│   │
│   ├── 💾 database/                 # Data persistence
│   │   ├── chroma_setup.py          #   ChromaDB vector store setup/load
│   │   ├── persistent_db.py         #   SQLite — sessions, chat history, LangChain adapter
│   │   ├── session_db.py            #   In-memory session cache
│   │   ├── chroma_store/            #   ChromaDB data files (gitignored)
│   │   └── sessions.db              #   SQLite database file (gitignored)
│   │
│   └── 🧪 testing/
│       └── list_collections.py      #   Debug utility for ChromaDB collections
│
├── 📸 assets/                       # Screenshots for README
│
└── 📂 database/
    └── files/                       # Temporary file storage for Excel processing
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description | Body |
|--------|----------|-------------|------|
| `POST` | `/summarize` | Summarize one or more PDFs | `files`: multipart PDF(s) |
| `POST` | `/qa` | Ask a question across PDFs | `files`: PDFs, `question`: str, `session_id`: str |
| `POST` | `/process_excel` | Query an Excel file | `file`: Excel, `query`: str, `session_id`: str |
| `GET` | `/sessions` | List all past sessions | — |
| `GET` | `/history/<session_id>` | Get chat history for a session | — |

---

## 🧩 How the RAG Pipeline Works

Understanding the RAG (Retrieval-Augmented Generation) pipeline is key to understanding this project:

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant API as Flask API
    participant PP as PreProcessors
    participant VDB as ChromaDB
    participant LLM as Gemini 3.5 Flash
    participant DB as SQLite

    Note over User,DB: 📥 Phase 1 — Document Ingestion
    User->>UI: Upload PDF files
    UI->>API: POST /summarize (files)
    API->>PP: Extract text (PyPDF2)
    PP-->>API: Raw text
    API->>LLM: Summarize text
    LLM-->>API: Summary
    API-->>UI: Display summaries

    Note over User,DB: ❓ Phase 2 — Question Answering
    User->>UI: Ask question
    UI->>API: POST /qa (files, question, session_id)
    API->>PP: Extract text → Chunk (2500 chars)
    PP-->>API: Document chunks
    API->>VDB: Embed & store chunks
    
    Note over API,LLM: History-aware reformulation
    API->>DB: Load chat history
    DB-->>API: Past messages
    API->>LLM: Reformulate question with context
    LLM-->>API: Standalone question
    
    API->>VDB: Semantic search (top-5)
    VDB-->>API: Relevant chunks
    API->>LLM: Answer with context
    LLM-->>API: Answer + sources
    
    API->>LLM: Generate 3 follow-up suggestions
    LLM-->>API: Suggestions
    
    API->>DB: Save Q&A to chat history
    API-->>UI: Answer, sources, suggestions
    UI-->>User: Display conversation
```

### Key Design Decisions

1. **History-Aware Retriever** — Before searching the vector store, the user's question is reformulated using chat history. This means follow-up questions like *"What about section 3?"* are understood in context.

2. **Per-Session Vector Collections** — Each session gets its own ChromaDB collection (`langchain_{session_id}`), so different document sets don't contaminate each other's search results.

3. **Dual Persistence** — In-memory cache for speed during active sessions + SQLite for permanent storage that survives server restarts.

4. **Source-Tagged Chunks** — Every text chunk carries its source filename in metadata, enabling precise citation back to the original document.

---

## 📝 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ | Your Google AI API key for Gemini access |

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

<div align="center">

**Built with ❤️ by [Ishmeet Singh](https://github.com/Ishmeet04)**

*If this project helped you, give it a ⭐ on GitHub!*

</div>
