


import logging
import os
from flask import Flask, request, jsonify
from io import BytesIO
from backend.services.csv_qa import process_excel_for_qa
from backend.services.summary_service import summarize_pdf, summarize_multiple_pdfs
from backend.database.session_db import get_session_data
from backend.database.persistent_db import get_all_sessions
from backend.services.q_a import q_a
from dotenv import load_dotenv  

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))


app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

# Ensure the folder structure exists
UPLOAD_FOLDER = os.path.join('database', 'files')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/summarize', methods=['POST'])
def summarize():
    app.logger.debug("Summarize request received")
    uploaded_files = request.files.getlist('files')
    if not uploaded_files or all(f.filename == '' for f in uploaded_files):
        return jsonify({'error': 'No files provided'}), 400

    files = [(BytesIO(f.read()), f.filename) for f in uploaded_files if f.filename]
    if len(files) == 1:
        # Single file: return a simple summary string for backward compat
        summary = summarize_pdf(files[0][0])
        return jsonify({'summary': summary, 'summaries': {files[0][1]: summary}})
    else:
        # Multiple files: return per-file summaries
        summaries = summarize_multiple_pdfs(files)
        return jsonify({'summaries': summaries})

@app.route('/qa', methods=['POST'])
def qa():
    app.logger.debug("Q&A Request received")
    if 'question' not in request.form or 'session_id' not in request.form:
        return jsonify({'error': 'question and session_id are required'}), 400

    uploaded_files = request.files.getlist('files')
    if not uploaded_files or all(f.filename == '' for f in uploaded_files):
        return jsonify({'error': 'At least one PDF file is required'}), 400

    question = request.form['question']
    session_id = request.form['session_id']

    files = [(BytesIO(f.read()), f.filename) for f in uploaded_files if f.filename]
    app.logger.debug(f"Files received: {[name for _, name in files]}")

    result = q_a(files, question, session_id)
    app.logger.debug(f"Answer: {result['answer']}")
    return jsonify({
        'answer':      result['answer'],
        'sources':     result['sources'],
        'suggestions': result['suggestions'],
        'chat_history': result['chat_history'],
    })

@app.route('/sessions', methods=['GET'])
def sessions():
    """Return all past sessions for the resume-session UI."""
    return jsonify({'sessions': get_all_sessions()})

@app.route('/history/<session_id>', methods=['GET'])
def history(session_id):
    """Return full chat history for a session from SQLite."""
    from backend.database.persistent_db import get_messages_as_dicts
    msgs = get_messages_as_dicts(session_id)
    return jsonify({'history': msgs})

@app.route('/process_excel', methods=['POST'])
def process_excel():
    app.logger.debug("Excel processing request received")
    if 'file' not in request.files:
        app.logger.debug("No file part in the request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        app.logger.debug("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    app.logger.debug(f"File saved to {file_path}")

    if 'query' not in request.form or 'session_id' not in request.form:
        return jsonify({'error': 'Query and session_id are required'}), 400

    query = request.form['query']
    session_id = request.form['session_id']

    response_text = process_excel_for_qa(file_path, query, session_id)
    
    return jsonify({'message': response_text}), 200

if __name__ == '__main__':
    app.run(debug=True)

