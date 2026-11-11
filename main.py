import json
from pathlib import Path

import gradio as gr
import requests
from PyPDF2 import PdfReader
from docx import Document

# Ollama API configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"


def extract_text(file_path):
    """Extract text from a PDF, DOCX, or TXT file."""
    if not file_path:
        return "Please upload a document."

    filename = Path(file_path)
    extension = filename.suffix.lower()

    try:
        if extension == ".pdf":
            reader = PdfReader(str(filename))
            pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            text = "\n".join(pages)

        elif extension == ".docx":
            doc = Document(str(filename))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs)

        elif extension == ".txt":
            text = filename.read_text(encoding="utf-8")

        else:
            return "Unsupported file type. Please upload a PDF, DOCX, or TXT file."

        if not text.strip():
            return "No readable text was found in the document."

        return text.strip()

    except Exception as exc:
        return f"Error extracting document text: {exc}"


def ollama_generate(prompt, max_tokens=1000):
    """Send a prompt to Ollama and collect its streamed response."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=(10, 300),
        )
        response.raise_for_status()

        output = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
                if "response" in data:
                    output.append(data["response"])
                if data.get("done", False):
                    break
            except json.JSONDecodeError:
                continue

        result = "".join(output).strip()
        return result if result else "The model returned no response."

    except requests.exceptions.ConnectionError:
        return (
            "Could not connect to Ollama. Make sure Ollama is running "
            "and the model is downloaded."
        )
    except requests.exceptions.Timeout:
        return "Ollama request timed out. Try a smaller document."
    except requests.exceptions.RequestException as exc:
        return f"Ollama API error: {exc}"


def analyze_document(file_path, user_question=""):
    """Summarize the uploaded document and optionally answer a question."""
    text = extract_text(file_path)

    if text.startswith(("Please upload", "Unsupported", "No readable", "Error extracting")):
        return text

    # Keep prompt size manageable for a small local model.
    max_chars = 24000
    document_context = text[:max_chars]
    if len(text) > max_chars:
        document_context += "\n\n[Document text truncated due to length.]"

    summary_prompt = f"""
You are an AI document analyzer. Read the document and produce a clear,
structured summary. Use headings and bullet points where appropriate.
Identify the main purpose, important points, key terms, and conclusion.
Do not invent facts. If the document is a contract, note parties, payment,
obligations, confidentiality, liability, and termination where present.
If it is a literary work, note characters, plot, themes, and moral where present.

DOCUMENT:
{document_context}
"""
    summary = ollama_generate(summary_prompt, max_tokens=800)

    if user_question and user_question.strip():
        qa_prompt = f"""
You are a document question-answering assistant.
Answer the question only using the supplied document. If the answer is not
available in the document, say that the document does not provide the answer.
Do not make up information.

DOCUMENT:
{document_context}

QUESTION:
{user_question.strip()}
"""
        answer = ollama_generate(qa_prompt, max_tokens=500)
        return (
            f"--- SUMMARY ---\n{summary}\n\n"
            f"--- QUESTION & ANSWER ---\n"
            f"Question: {user_question.strip()}\nAnswer: {answer}"
        )

    return f"--- SUMMARY ---\n{summary}"


with gr.Blocks(title="AI Document Analyzer") as demo:
    gr.Markdown("# AI Document Analyzer")
    gr.Markdown(
        "Upload a PDF, Word document (.docx), or text file (.txt) to generate "
        "a structured summary with Qwen 2.5 through Ollama. You may also ask "
        "a question about the document."
    )

    with gr.Row():
        file_input = gr.File(
            label="Upload Document",
            file_types=[".pdf", ".docx", ".txt"],
            type="filepath",
        )
        question_input = gr.Textbox(
            label="Optional Question",
            placeholder="Ask a question about the uploaded document...",
            lines=3,
        )

    analyze_button = gr.Button("Analyze Document", variant="primary")
    output = gr.Textbox(
    label="Summary and Q&A",
    lines=22,
)

    analyze_button.click(
        fn=analyze_document,
        inputs=[file_input, question_input],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
