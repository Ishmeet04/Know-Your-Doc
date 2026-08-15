import streamlit as st
import requests
import uuid

st.set_page_config(page_title="Know Your Doc", layout="wide")

# ── Session state init ───────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def reset_session():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.uploaded_filenames = []
    st.session_state.summaries = {}
    st.session_state.pop('vectorstore', None)
    st.session_state.pop('uploaded_file', None)


# ── Pages ────────────────────────────────────────────────────────────────────

def show_main_page():
    st.title('Know Your Doc')
    st.markdown("#### Choose the type of file you want to work with:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Work with PDF', use_container_width=True):
            st.session_state.page = 'pdf'
            st.rerun()
    with col2:
        if st.button('Work with Excel', use_container_width=True):
            st.session_state.page = 'excel'
            st.rerun()

    # ── Past sessions sidebar ────────────────────────────────────────────────
    st.divider()
    st.markdown("### Resume a Past Session")
    try:
        resp = requests.get('http://localhost:5001/sessions', timeout=3)
        if resp.status_code == 200:
            past = resp.json().get('sessions', [])
            if not past:
                st.info("No past sessions found.")
            for s in past[:8]:
                sid = s['session_id']
                fnames = ", ".join(s['filenames']) or "unknown files"
                ts = s['created_at'][:16].replace('T', ' ')
                label = f"📂 {fnames}  ·  {ts}"
                if st.button(label, key=f"resume_{sid}"):
                    st.session_state.session_id = sid
                    st.session_state.uploaded_filenames = s['filenames']
                    st.session_state.chat_history = []   # will load from DB on first ask
                    st.session_state.summaries = {}
                    st.session_state.page = 'pdf'
                    st.rerun()
    except Exception:
        st.caption("(Past sessions unavailable — Flask may not be running)")


def show_pdf_page():
    st.subheader('PDF Summarizer & Q&A')

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type="pdf",
        key="pdf_uploader",
        accept_multiple_files=True,
    )

    if not uploaded_files:
        # Check if this is a resumed session (filenames were pre-set from main page)
        resumed_filenames = st.session_state.get('uploaded_filenames', [])
        session_id = st.session_state.session_id

        if resumed_filenames:
            # ── Resumed session — show past context ──────────────────────────
            st.info(
                f"**Resumed session** — previously used: "
                + ", ".join(f"`{f}`" for f in resumed_filenames)
                + "\n\nRe-upload the same file(s) above to continue asking questions."
            )

            # Load past conversation from SQLite
            try:
                r = requests.get(f'http://localhost:5001/history/{session_id}', timeout=3)
                if r.status_code == 200:
                    past_msgs = r.json().get('history', [])
                    if past_msgs:
                        st.markdown("#### Past Conversation")
                        for msg in past_msgs:
                            role = "user" if msg['role'] == 'human' else "assistant"
                            with st.chat_message(role):
                                st.write(msg['content'])
                                if msg.get('timestamp'):
                                    st.caption(msg['timestamp'])
                    else:
                        st.caption("No past messages found for this session.")
            except Exception:
                st.caption("Could not load past history — Flask may not be running.")
        else:
            st.info("Upload one or more PDFs to get started.")

        if st.button('← Back'):
            st.session_state.page = 'main'
            st.rerun()
        return

    # Detect file set change → reset session
    current_filenames = sorted([f.name for f in uploaded_files])
    if st.session_state.get('uploaded_filenames') != current_filenames:
        reset_session()
        st.session_state.uploaded_filenames = current_filenames

        # AUTO-SUMMARIZE when a new file set is uploaded
        with st.spinner(f"Auto-summarizing {len(uploaded_files)} PDF(s)..."):
            for f in uploaded_files:
                f.seek(0)
            fp = [('files', (f.name, f.read(), 'application/pdf')) for f in uploaded_files]
            r = requests.post('http://localhost:5001/summarize', files=fp)
        if r.status_code == 200:
            st.session_state.summaries = r.json().get('summaries', {})
        else:
            st.warning("Auto-summarization failed.")

    # File list
    st.markdown(f"**{len(uploaded_files)} file(s) uploaded:**")
    for f in uploaded_files:
        st.markdown(f"- `{f.name}` ({f.size / 1024:.1f} KB)")
    st.divider()

    # ── Summaries ────────────────────────────────────────────────────────────
    st.markdown("### Summaries")
    if st.button('Re-summarize All PDFs', use_container_width=True):
        with st.spinner("Summarizing..."):
            for f in uploaded_files:
                f.seek(0)
            fp = [('files', (f.name, f.read(), 'application/pdf')) for f in uploaded_files]
            r = requests.post('http://localhost:5001/summarize', files=fp)
        if r.status_code == 200:
            st.session_state.summaries = r.json().get('summaries', {})

    if st.session_state.get('summaries'):
        for filename, summary in st.session_state.summaries.items():
            with st.expander(f"Summary: {filename}", expanded=True):
                st.write(summary)
    st.divider()

    # ── Q&A ──────────────────────────────────────────────────────────────────
    st.markdown("### Ask Questions Across All PDFs")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pending_question' not in st.session_state:
        st.session_state.pending_question = ''

    question = st.text_input(
        "Ask a question about your documents:",
        value=st.session_state.pending_question,
        key="qa_input",
    )

    col_ask, col_clear = st.columns([3, 1])
    ask_clicked = col_ask.button('Ask', use_container_width=True)
    if col_clear.button('Clear Chat', use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching across documents..."):
                for f in uploaded_files:
                    f.seek(0)
                fp = [('files', (f.name, f.read(), 'application/pdf')) for f in uploaded_files]
                dp = {'question': question, 'session_id': st.session_state.session_id}
                r = requests.post('http://localhost:5001/qa', files=fp, data=dp)

            if r.status_code == 200:
                data = r.json()
                answer      = data.get('answer', '')
                sources     = data.get('sources', [])
                suggestions = data.get('suggestions', [])

                st.session_state.chat_history.append({
                    'question':    question,
                    'answer':      answer,
                    'sources':     sources,
                    'suggestions': suggestions,
                })
                st.session_state.pending_question = ''
            else:
                st.error(f"Failed to get answer: {r.status_code}")

    # ── Conversation display ─────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("#### Conversation")
        for chat in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(chat['question'])
            with st.chat_message("assistant"):
                st.write(chat['answer'])
                # Source citations
                if chat.get('sources'):
                    src_list = "  ·  ".join(f"📄 `{s}`" for s in chat['sources'])
                    st.caption(f"Sources: {src_list}")
                # Follow-up suggestion chips
                if chat.get('suggestions'):
                    st.markdown("**Suggested follow-ups:**")
                    cols = st.columns(len(chat['suggestions']))
                    for i, suggestion in enumerate(chat['suggestions']):
                        if cols[i].button(suggestion, key=f"sug_{id(chat)}_{i}"):
                            st.session_state.pending_question = suggestion
                            st.rerun()

    st.divider()
    if st.button('← Back to Home'):
        st.session_state.page = 'main'
        st.rerun()


def show_excel_page():
    st.subheader('Excel Processor')
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xls", "xlsx"])
    query = st.text_input("Enter your query:")
    if uploaded_file is not None and query:
        st.write("Uploaded File:", uploaded_file.name)
        if st.button('Process Excel'):
            files = {'file': uploaded_file}
            data = {'query': query, 'session_id': st.session_state.session_id}
            response = requests.post('http://localhost:5001/process_excel', files=files, data=data)
            if response.status_code == 200:
                data = response.json().get('message')
                st.write('Processed Data:', data)
            else:
                st.write("Failed to process Excel file:", response.status_code)
    if st.button('Go Back'):
        st.session_state.page = 'main'
        st.rerun()


# ── Router ───────────────────────────────────────────────────────────────────
if st.session_state.page == 'main':
    show_main_page()
elif st.session_state.page == 'pdf':
    show_pdf_page()
elif st.session_state.page == 'excel':
    show_excel_page()


# Initialize session state if not already set
if 'page' not in st.session_state:
    st.session_state.page = 'main'

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def reset_session():
    st.session_state.vectorstore = None
    st.session_state.chat_history = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.uploaded_file = None
    st.session_state.uploaded_filenames = []
    st.session_state.summaries = {}
