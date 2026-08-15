

# from langchain_text_splitters import CharacterTextSplitter
# from langchain.docstore.document import Document

# class TextToChunks:
#     def __init__(self, text):
#         self.text = text

#     def split_text(self):
#         text_splitter = CharacterTextSplitter(
#             separator="\n",
#             chunk_size=2500,
#             chunk_overlap=200,
#             length_function=len,
#             is_separator_regex=False,
#         )
#         chunks = text_splitter.split_text(self.text)
#         # Ensure each chunk has a unique ID
#         documents = [Document(page_content=chunk, metadata={"id": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]
#         return documents
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

class TextToChunks:
    def __init__(self, text, source: str = "unknown"):
        self.text = text
        self.source = source

    def split_text(self):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2500,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_text(self.text)
        # Tag every chunk with its source filename for citation in answers
        documents = [
            Document(page_content=chunk, metadata={"source": self.source})
            for chunk in chunks
        ]
        return documents
