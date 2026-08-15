import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def handle_csv_query(csv_path, query):
    model = genai.GenerativeModel("gemini-3.5-flash")
    sample_file = genai.upload_file(path=csv_path, display_name="test")

    response = model.generate_content([sample_file, query])
    result = response.text

    genai.delete_file(sample_file.name)
    return result
