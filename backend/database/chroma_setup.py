from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import logging

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

# Persistent storage directory for ChromaDB
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)


def setup_chroma_vector_db(chunks, session_id):
    """Create a new ChromaDB collection from document chunks and persist to disk."""
    collection_name = f"langchain_{session_id}"
    logging.debug(f"Creating Chroma collection '{collection_name}' with {len(chunks)} chunks")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    logging.debug("Chroma vectorstore created and persisted")
    return vectorstore


def load_chroma_vector_db(session_id):
    """
    Load an existing ChromaDB collection from disk.
    Returns None if the collection doesn't exist or is empty.
    Used for session resume after server restart.
    """
    collection_name = f"langchain_{session_id}"
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        count = vectorstore._collection.count()
        if count == 0:
            return None
        logging.debug(f"Loaded existing Chroma collection '{collection_name}' ({count} chunks)")
        return vectorstore
    except Exception as e:
        logging.debug(f"Could not load Chroma collection '{collection_name}': {e}")
        return None