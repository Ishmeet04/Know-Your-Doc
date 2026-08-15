# Ensure this file contains something like this:


# from backend.models.gemini import google_llm  # Correct import path

from langchain_core.prompts import PromptTemplate  # Corrected import as per warning
from langchain_core.output_parsers import StrOutputParser


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

google_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])
def get_summarization_chain():
    llm = google_llm
    prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    return (
        prompt       # Prompt for Gemini
        | llm        # Gemini function
        | StrOutputParser()  # Output parser
    )
