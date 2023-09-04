import os
import PyPDF2
import openai
import gradio as gr

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

def ai_tool(user_input, pdf_text):
    # Calculate the remaining tokens for the completion
    max_tokens_limit = 4097 - len(user_input)

    # Truncate the PDF text to fit within the model's maximum context length
    pdf_text_truncated = pdf_text[:max_tokens_limit]

    # Prepend the user's query and truncated PDF text to form the prompt
    prompt = f"User Query: {user_input}\nPDF Text: {pdf_text_truncated}\n"

    # Set up OpenAI API credentials (replace 'YOUR_API_KEY' with your actual API key)
    openai.api_key = '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'

    # Use GPT-3 to generate a response based on the prompt
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100 #changed to be smaller
    )

    return response.choices[0].text.strip()


def main(user_input):
    pdf_folder = '/Users/Elise/Documents/PAPAI/docs'

    # Collect all PDF files in the specified folder
    pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith('.pdf')]

    # Initialize a dictionary to store the AI tool responses
    ai_responses = {}

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        pdf_text = extract_text_from_pdf(pdf_path)

        # Use the AI tool to generate a response
        response = ai_tool(user_input, pdf_text)
        ai_responses[pdf_file] = response

    return ai_responses

# Define the Gradio interface
iface = gr.Interface(
    fn=main,
    inputs=gr.inputs.Textbox(label="Enter your query here"),
    outputs=gr.outputs.JSON(label="Results"),
    live=True,
    capture_session=True,
    title="AI PDF Text Referencing Tool",
    description="Enter your query, and the AI tool will reference the PDF documents to provide relevant information.",
)

if __name__ == "__main__":
    iface.launch()


#The ai_tool(user_input, pdf_text) function uses GPT-3 to generate a response based
# on the user's input query and the truncated PDF text. The AI response is then returned.
#
#The main(user_input) function is the core of the tool. It reads all PDF files in a 
# specified folder, extracts their text, and uses the ai_tool function to generate AI
# responses for each PDF file based on the user's query. The responses are stored in a 
# dictionary where the PDF filenames are keys, and the AI-generated responses are values.