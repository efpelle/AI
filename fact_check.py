import os
import PyPDF2
import openai
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
import nltk
from nltk.metrics import jaccard_distance

# Define constants
API_KEY = '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'  # Your API key
PDF_FOLDER = '/Users/Elise/Documents/PAPAI/docs'
MAX_TOKENS_LIMIT = 4097
BATCH_SIZE = 5

# Initialize OpenAI API
openai.api_key = API_KEY

# Initialize NLTK
nltk.download("punkt")

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return "".join(page.extract_text() for page in reader.pages)

def ai_tool(user_input, pdf_text):
    # Create a prompt for GPT-3
    prompt = f"{user_input}\n{pdf_text}\n"

    # Generate response using GPT-3
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    
    generated_text = response.choices[0].text.strip()
    
    # Perform fact-checking before returning the response
    if is_factually_correct(generated_text):
        return generated_text
    else:
        return "No accurate information was found."

# Adjusted fact-checking function with a slightly lower threshold
def is_factually_correct(response):
    trusted_source = "The official website of XYZ Organization."  # Example trusted source

    # Calculate Jaccard similarity between response and trusted source
    response_tokens = set(nltk.word_tokenize(response.lower()))
    trusted_tokens = set(nltk.word_tokenize(trusted_source.lower()))
    jaccard_sim = 1 - jaccard_distance(response_tokens, trusted_tokens)

    # Adjust this threshold based on your desired strictness level
    accuracy_threshold = 0.2  # Adjusted value for slight looseness

    return jaccard_sim >= accuracy_threshold

def process_pdf(user_input, pdf_file):  # Add user_input as an argument
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    pdf_text = extract_text_from_pdf(pdf_path)[:MAX_TOKENS_LIMIT]
    response_text = ai_tool(user_input, pdf_text)
    return pdf_file, response_text

def main(user_input):
    pdf_files = [file for file in os.listdir(PDF_FOLDER) if file.endswith('.pdf')]

    excluded_phrases = ["no relevant information", "not applicable", "irrelevant data"]

    ai_responses = {}
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda pdf_file: process_pdf(user_input, pdf_file), pdf_files))

    user_input_words = set(user_input.lower().split())  # Convert input to lowercase words

    for pdf_file, response_text in results:
        response_lower = response_text.lower()
        
        if response_text and \
           not any(phrase in response_lower for phrase in excluded_phrases) and \
           "no information was found" not in response_lower and \
           len(user_input_words.intersection(response_lower.split())) >= 2:
            ai_responses[pdf_file] = response_text
    
    return ai_responses

iface = gr.Interface(
    fn=main,
    inputs=gr.inputs.Textbox(label="Enter your query here"),
    outputs=gr.outputs.JSON(label="Results"),
    live=True,
    capture_session=True,
    title="iParametrcs AI Public Appeals Referencing Tool",
    description="Enter your query, and the AI tool will reference the FEMA Public Appeals documents to provide relevant information.",
)

if __name__ == "__main__":
    iface.launch()
