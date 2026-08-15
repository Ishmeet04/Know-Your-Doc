from backend.PreProcessors.pdf_to_text import PDFToText
from backend.PreProcessors.textToChunks import TextToChunks
from backend.database.chroma_setup import setup_chroma_vector_db, load_chroma_vector_db
from backend.database.session_db import get_session_data, set_session_data
from backend.database.persistent_db import save_session, get_messages_as_dicts
from backend.models.qa_RAG import create_rag_chain, invoke_rag_chain, generate_followup_suggestions
from io import BytesIO
import logging


def q_a(files: list, question: str, session_id: str) -> dict:
    """
    files: list of (file_content: BytesIO, filename: str) tuples
    Returns: {answer, sources, suggestions, chat_history}
    """
    logging.debug(f"Session ID: {session_id}")
    session_data = get_session_data(session_id)

    incoming_filenames = sorted([name for _, name in files])

    needs_rebuild = (
        not session_data
        or "vectorstore" not in session_data
        or session_data.get("filenames") != incoming_filenames
    )

    if needs_rebuild:
        # Try loading from disk first (session resume after restart)
        vectorstore = load_chroma_vector_db(session_id)

        if vectorstore is not None and session_data and session_data.get("filenames") == incoming_filenames:
            logging.debug("Resumed vectorstore from disk")
        else:
            # Re-embed all PDFs
            logging.debug(f"Building vectorstore for: {incoming_filenames}")
            all_chunks = []
            for file_content, filename in files:
                extractor = PDFToText(file_content)
                extracted_text = extractor.extract_text()
                chunker = TextToChunks(extracted_text, source=filename)
                all_chunks.extend(chunker.split_text())

            vectorstore = setup_chroma_vector_db(all_chunks, session_id)
            logging.debug(f"Vectorstore built with {len(all_chunks)} chunks")

        set_session_data(session_id, {"vectorstore": vectorstore, "filenames": incoming_filenames})
        save_session(session_id, incoming_filenames)
    else:
        logging.debug("Reusing in-memory vectorstore")
        vectorstore = session_data["vectorstore"]

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    rag_chain = create_rag_chain(retriever)
    result = invoke_rag_chain(rag_chain, question, session_id)

    answer = result["answer"]
    sources = result["sources"]

    # Generate follow-up suggestions (non-blocking — empty list on failure)
    context_docs = retriever.invoke(question)
    suggestions = generate_followup_suggestions(answer, context_docs)

    # Load full persisted chat history for the frontend
    chat_history = get_messages_as_dicts(session_id)

    logging.debug(f"Answer: {answer} | Sources: {sources} | Suggestions: {suggestions}")
    return {
        "answer": answer,
        "sources": sources,
        "suggestions": suggestions,
        "chat_history": chat_history,
    }