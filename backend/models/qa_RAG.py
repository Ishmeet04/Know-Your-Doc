from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from backend.database.persistent_db import SQLiteChatMessageHistory
from dotenv import load_dotenv
import os
import json
import logging

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])


def create_rag_chain(retriever):
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return SQLite-backed chat history — persists across server restarts."""
    return SQLiteChatMessageHistory(session_id)


def invoke_rag_chain(rag_chain, question: str, session_id: str) -> dict:
    """
    Run the RAG chain and return:
      {
        "answer":  str,
        "sources": list[str],   # unique source filenames from retrieved docs
      }
    """
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    result = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )

    answer = result["answer"]

    # Extract unique source filenames from retrieved document metadata
    context_docs = result.get("context", [])
    sources = sorted(set(
        doc.metadata.get("source", "Unknown")
        for doc in context_docs
        if doc.metadata.get("source")
    ))

    return {"answer": answer, "sources": sources}


def generate_followup_suggestions(answer: str, context_docs: list) -> list:
    """
    Given an answer and its source documents, ask the LLM to suggest
    3 short follow-up questions the user might want to ask next.
    Returns a list of up to 3 question strings.
    """
    context_snippet = " ".join(
        doc.page_content[:300] for doc in context_docs[:3]
    )
    prompt = f"""Based on this answer and the document context below, suggest exactly 3 short \
follow-up questions a user might ask next. Return ONLY a JSON array of 3 strings, no explanation.

Answer: {answer}

Context snippet: {context_snippet}"""
    try:
        response = llm.invoke(prompt)
        # Strip markdown fences if present
        raw = response.content.strip().strip("```json").strip("```").strip()
        suggestions = json.loads(raw)
        if isinstance(suggestions, list):
            return [str(s) for s in suggestions[:3]]
    except Exception as e:
        logging.warning(f"Follow-up suggestion failed: {e}")
    return []