# from backend.PreProcessors.pdf_to_text import PDFToText
# from backend.models.summarization_chain import get_summarization_chain

# def summarize_pdf(file_path):
#     # Load PDF and extract text
#     extractor = PDFToText(file_path)
#     extracted_text = extractor.extract_text()

#     # Get the summarization chain and invoke it
#     summarization_chain = get_summarization_chain()
#     return summarization_chain.invoke(extracted_text)




from backend.PreProcessors.pdf_to_text import PDFToText
from backend.models.summarization_chain import get_summarization_chain
from io import BytesIO


def summarize_pdf(file_content):
    """Summarize a single PDF (BytesIO)."""
    extractor = PDFToText(file_content)
    extracted_text = extractor.extract_text()
    summarization_chain = get_summarization_chain()
    return summarization_chain.invoke(extracted_text)


def summarize_multiple_pdfs(files: list) -> dict:
    """
    Summarize multiple PDFs individually.
    files: list of (file_content: BytesIO, filename: str)
    Returns: {filename: summary_string}
    """
    summaries = {}
    chain = get_summarization_chain()
    for file_content, filename in files:
        extractor = PDFToText(file_content)
        text = extractor.extract_text()
        summaries[filename] = chain.invoke(text)
    return summaries
