import os
import numpy as np
import faiss
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from werkzeug.utils import secure_filename
import uuid
from flask import Response, stream_with_context, send_from_directory
import time
import json
import google.generativeai as genai


load_dotenv()
os.environ["GENAI_API_VERIFY_SSL"] = "false"
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'documents'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf', 'txt'}


FAISS_INDEX_PATH = "faiss_index.idx"
METADATA_PATH = "metadata.pkl"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)





FAISS_INDEX = None
DOC_CHUNKS = []
METADATA = []

AI_MODEL = "gemini-2.5-flash"

model = genai.GenerativeModel(AI_MODEL)



SYSTEM_PROMPT = """
You are a specialized AI assistant for a hospital.
Your primary function is to answer questions based on the hospital's internal policy documents.
However, you can also engage in general conversation and use knowledge outside the contexts whenever necessary.

Organization Information (Context):
- The organization is a hospital located in Illinois, USA with Over 4000 employees.
- All advice, interpretations, and suggestions must strictly adhere to HIPAA regulations and general healthcare compliance standards.
- When answering from documents and be precise.
- When engaging in general conversation, be helpful and polite.
"""


# REPLACE this entire function in your app.py

def build_or_load_index(force_rebuild=False):
    """
    Builds a new FAISS index from documents in the UPLOAD_FOLDER or loads an existing one.
    CORRECTED to use the new genai.embed_content function.
    """
    global FAISS_INDEX, DOC_CHUNKS, METADATA
    print("\n--- Running build_or_load_index ---")

    if force_rebuild and os.path.exists(FAISS_INDEX_PATH):
        print("Forcing rebuild: Deleting old index files.")
        if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
        if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        print(f"Loading existing index from {FAISS_INDEX_PATH}...")
        FAISS_INDEX = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            data = pickle.load(f)
            DOC_CHUNKS = data["chunks"]
            METADATA = data["metadata"]
        print(f"Index loaded successfully. Total vectors: {FAISS_INDEX.ntotal}")
        return

    print("No existing index found. Building a new one...")
    all_chunks = []
    all_metadata = []

    doc_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    print(f"Found {len(doc_files)} documents to index: {doc_files}")

    if not doc_files:
        print("No documents found in the upload folder. Index will be empty.")
        FAISS_INDEX, DOC_CHUNKS, METADATA = None, [], []
        return

    for doc_file in doc_files:
        path = os.path.join(app.config['UPLOAD_FOLDER'], doc_file)
        print(f" > Processing {path}...")
        try:
            if doc_file.endswith('.pdf'):
                reader = PdfReader(path)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        paragraphs = text.split('\n\n')
                        for para in paragraphs:
                            if len(para.strip()) > 50:
                                all_chunks.append(para.strip())
                                all_metadata.append({"source": doc_file, "page": page_num + 1})
            elif doc_file.endswith('.txt'):
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    paragraphs = text.split('\n\n')
                    for para in paragraphs:
                        if len(para.strip()) > 50:
                            all_chunks.append(para.strip())
                            all_metadata.append({"source": doc_file, "page": 1})
        except Exception as e:
            print(f"   [ERROR] Failed to process {doc_file}: {e}")

    if not all_chunks:
        print("No valid text chunks were extracted. Index cannot be built.")
        FAISS_INDEX, DOC_CHUNKS, METADATA = None, [], []
        return

    DOC_CHUNKS = all_chunks
    METADATA = all_metadata

    print(f"Generated {len(DOC_CHUNKS)} total chunks. Now generating embeddings...")
    # THE FIX: Use the new, correct embed_content function
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=DOC_CHUNKS,
        task_type="RETRIEVAL_DOCUMENT"
    )

    # THE FIX: Access the results from the response dictionary
    embeddings = response['embedding']
    embedding_matrix = np.array(embeddings, dtype='float32')

    embedding_dimension = embedding_matrix.shape[1]
    FAISS_INDEX = faiss.IndexFlatL2(embedding_dimension)
    FAISS_INDEX.add(embedding_matrix)

    print(f"Saving index and metadata to {FAISS_INDEX_PATH}...")
    faiss.write_index(FAISS_INDEX, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump({"chunks": DOC_CHUNKS, "metadata": METADATA}, f)
    print(f"Index built and saved successfully. Total vectors: {FAISS_INDEX.ntotal}")

# def answer_query(question, selected_docs, k=8):  # Increased k for better analysis context
#     from google import genai
#     client = genai.Client(api_key=API_KEY)
#     ANALYTICAL_QUESTIONS = [
#         "Identify gaps or missing policies?",
#         "Highlight non-compliance or outdated areas?",
#         "Compare to industry best practices or standards?",
#         "Suggest changes or improvements?",
#     ]
#
#
#     greetings = ["hi", "hello", "hey", "how are you"]
#     if question.lower().strip() in greetings:
#         prompt = f"{SYSTEM_PROMPT}\n\nThe user just said '{question}'. Respond politely and briefly."
#         response = client.models.generate_content(model=AI_MODEL, contents=prompt)
#         return {"answer": response.text, "sources": [], "follow_up": ANALYTICAL_QUESTIONS}
#
#     if FAISS_INDEX is None or FAISS_INDEX.ntotal == 0:
#         return {"answer": "The document index is empty. Please upload documents first.", "sources": [], "follow_up": []}
#
#     # --- RAG Context Retrieval (same for both query types) ---
#     response = client.models.embed_content(model="text-embedding-004", contents=[question])
#     query_embedding = np.array(response.embeddings[0].values, dtype='float32').reshape(1, -1)
#
#     # We retrieve context regardless, as even analytical questions need it
#     distances, indices = FAISS_INDEX.search(query_embedding, FAISS_INDEX.ntotal)
#
#     filtered_results = []
#     # A looser threshold for analytical questions to gather broader context
#     distance_threshold = 1.5 if question in ANALYTICAL_QUESTIONS else 1.2
#
#     for i, dist in zip(indices[0], distances[0]):
#         if METADATA[i]['source'] in selected_docs and dist < distance_threshold:
#             filtered_results.append({'index': i})
#
#     if not filtered_results:
#         # Fallback to conversational response if no context is found
#         prompt = f"{SYSTEM_PROMPT}\n\nThe user asked: '{question}'. There are no relevant policy documents found. Answer as a general AI assistant."
#         response = client.models.generate_content(model=AI_MODEL, contents=prompt)
#         return {"answer": response.text, "sources": [], "follow_up": ANALYTICAL_QUESTIONS}
#
#     # --- Prompt Routing: Choose the right prompt for the job ---
#     top_k_indices = [res['index'] for res in filtered_results[:k]]
#     docs = [DOC_CHUNKS[i] for i in top_k_indices]
#     sources = sorted(list(set([f"{METADATA[i]['source']} (Page {METADATA[i]['page']})" for i in top_k_indices])))
#     context = "\n\n---\n\n".join(docs)
#
#     prompt_to_use = ""
#
#     if question in ANALYTICAL_QUESTIONS:
#         # If it's an analytical question, use the new, permissive prompt
#         print("Using ANALYTICAL prompt...")
#         prompt_to_use = f"""{SYSTEM_PROMPT}
#
# You are acting as an expert policy analyst and healthcare compliance consultant.
# Your task is to analyze the provided internal policy documents and answer the user's request.
# You MUST base your analysis on the provided context, but you are PERMITTED and ENCOURAGED to use your external knowledge of industry best practices, standards, and regulations (like HIPAA) to identify gaps, suggest improvements, or make comparisons.
#
# Context from policy documents:
# {context}
#
# User's Analytical Request:
# "{question}"
#
# Your detailed analysis:
# """
#     else:
#         # Otherwise, use the standard, strict RAG prompt
#         print("Using standard RAG prompt...")
#         prompt_to_use = f"""{SYSTEM_PROMPT}
#
# You MUST answer the question using ONLY the provided context.
# If the context is insufficient, state that you cannot answer from the provided documents. Do not use outside knowledge.
#
# Context from policy documents:
# {context}
#
# Question:
# {question}
#
# Answer:
# """
#
#     main_response = client.models.generate_content(model=AI_MODEL, contents=prompt_to_use)
#     answer = main_response.text
#
#     # Return the answer with the STATIC list of follow-up questions
#     return {"answer": answer, "sources": sources, "follow_up": ANALYTICAL_QUESTIONS}


#
# DELETE your old answer_query_stream function and REPLACE it with this one.
#
# REPLACE this entire function in your app.py

def answer_query_stream(question, selected_docs, k=8):
    """
    Complete and final streaming function, corrected to use the new
    genai.embed_content function for the query.
    """
    import json

    ANALYTICAL_QUESTIONS = [
        "Identify risks and policy gaps within the documents.",
        "Highlight possible non-compliance and risky areas in the documents.",
        "Compare if the following documents meets with industry best practices or standards.",
        "Suggest changes or improvements that can be made to the following documents.",
    ]

    greetings = ["hi", "hello", "hey", "how are you"]
    if question.lower().strip() in greetings:
        response_text = "Hello! How can I help you with your policy documents today?"
        if "how are you" in question.lower():
            response_text = "I'm just a program, but I'm ready to assist you!"
        yield json.dumps({"type": "token", "data": response_text}) + '\n'
        metadata = {"sources": [], "follow_up": ANALYTICAL_QUESTIONS}
        yield json.dumps({"type": "metadata", "data": metadata}) + '\n'
        return

    if FAISS_INDEX is None or FAISS_INDEX.ntotal == 0:
        error_message = "The document index is empty. Please upload documents first."
        yield json.dumps({"type": "token", "data": error_message}) + '\n'
        metadata = {"sources": [], "follow_up": []}
        yield json.dumps({"type": "metadata", "data": metadata}) + '\n'
        return

    # THE FIX: Use the new, correct embed_content function for the query
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=question,
        task_type="RETRIEVAL_QUERY"
    )

    # THE FIX: Access the single embedding and ensure it's a 2D array for FAISS
    query_embedding = np.array([response['embedding']], dtype='float32')

    distances, indices = FAISS_INDEX.search(query_embedding, FAISS_INDEX.ntotal)

    filtered_results = []
    distance_threshold = 1.5 if question in ANALYTICAL_QUESTIONS else 1.2
    for i, dist in zip(indices[0], distances[0]):
        if METADATA[i]['source'] in selected_docs and dist < distance_threshold:
            filtered_results.append({'index': i})

    prompt_to_use = ""
    sources = []

    if not filtered_results:
        prompt_to_use = f"{SYSTEM_PROMPT}\n\nThe user asked: '{question}'. There are no relevant policy documents found. Answer as a general AI assistant."
        sources = []
    else:
        top_k_indices = [res['index'] for res in filtered_results[:k]]
        docs = [DOC_CHUNKS[i] for i in top_k_indices]
        sources = sorted(list(set([f"{METADATA[i]['source']} (Page {METADATA[i]['page']})" for i in top_k_indices])))
        context = "\n\n---\n\n".join(docs)

        if question in ANALYTICAL_QUESTIONS:
            print("Using ANALYTICAL prompt...")
            prompt_to_use = f"""{SYSTEM_PROMPT}\n\nYou are acting as an expert policy analyst... (rest of your analytical prompt here)\n\nContext from policy documents:\n{context}\n\nUser's Analytical Request:\n"{question}"\n\nYour detailed analysis:"""
        else:
            print("Using standard RAG prompt...")
            prompt_to_use = f"""{SYSTEM_PROMPT}\n\nYou MUST answer the question using ONLY the provided context...\n\nContext from policy documents:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""

    print(f"\n>>> Generating content...")
    try:
        stream = model.generate_content(
            prompt_to_use,
            stream=True
        )

        for chunk in stream:
            if hasattr(chunk, 'text') and chunk.text:
                data = {"type": "token", "data": chunk.text}
                yield json.dumps(data) + '\n'

        print(">>> Stream finished. Yielding metadata.")
        final_metadata = {
            "type": "metadata",
            "data": {
                "sources": sources,
                "follow_up": ANALYTICAL_QUESTIONS
            }
        }
        yield json.dumps(final_metadata) + '\n'
    except Exception as e:
        print(f"\n--- ERROR DURING STREAM GENERATION: {e} ---\n")
        error_message = f"An error occurred with the AI model. Please check the server logs."
        data = {"type": "token", "data": error_message}
        yield json.dumps(data) + '\n'
        final_metadata = {"type": "metadata", "data": {"sources": [], "follow_up": []}}
        yield json.dumps(final_metadata) + '\n'
# --- Add placeholder functions if needed for copy-pasting ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Your Flask routes go here...
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/documents", methods=["GET"])
def get_documents():
    docs = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    return jsonify(sorted(docs))


@app.route("/upload", methods=["POST"])
def upload_document():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type or no file selected"}), 400
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    elif 'text_content' in request.form:
        text = request.form['text_content']
        if not text.strip():
            return jsonify({"error": "Text content cannot be empty"}), 400
        filename = f"text-{uuid.uuid4().hex}.txt"
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        return jsonify({"error": "No file or text content provided"}), 400

    build_or_load_index(force_rebuild=True)
    return jsonify({"success": f"'{filename}' uploaded and index rebuilt."})


def test_stream_generator():
    """
    A dead-simple generator that streams hardcoded data.
    This helps us test the connection without involving the AI model.
    """
    test_words = ["This ", "is ", "a ", "direct ", "test ", "from ", "the ", "server. "]

    # 1. Stream the text tokens
    for word in test_words:
        print(f"SERVER YIELDING: {word}")  # You will see this in your Flask terminal
        data = {"type": "token", "data": word}
        yield json.dumps(data) + '\n'
        time.sleep(0.1)  # Slow it down so we can see it stream

    # 2. Stream the final metadata
    print("SERVER YIELDING METADATA")
    metadata = {
        "sources": ["Test Source 1", "Test Source 2"],
        "follow_up": ["Did this test work?", "Go to next step"]
    }
    final_data = {"type": "metadata", "data": metadata}
    yield json.dumps(final_data) + '\n'
    print("SERVER FINISHED YIELDING")


# TEMPORARILY REPLACE YOUR /ask ROUTE WITH THIS ONE
# @app.route("/ask", methods=["POST"])
# def ask():
#     """
#     Temporary /ask route for debugging the stream connection.
#     """
#     print("\n--- /ask DEBUG ROUTE HIT ---")
#     return Response(stream_with_context(test_stream_generator()), content_type='application/x-ndjson')

@app.route("/ask", methods=["POST"])
def ask():
    """
    This is the new streaming version of your /ask route.
    """
    data = request.get_json(force=True)
    q = data.get("query")

    print(q)


    selected_docs = data.get("selected_docs", [])

    print(selected_docs)

    if not q:
        return jsonify({"error": "Query required"}), 400
    if not selected_docs:
        return jsonify({"error": "Please select at least one document to search."}), 400

    try:

        return Response(stream_with_context(answer_query_stream(q, selected_docs)),
                        content_type='application/x-ndjson')
    except Exception as e:
        import traceback
        traceback.print_exc()

        error_payload = json.dumps({"error": str(e)})
        return Response(error_payload, status=500, content_type='application/json')

@app.route("/review")
def review_page():
    """
    Renders the new PDF review page.
    """
    return render_template("review.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """
    Serves a specific file from the upload folder.
    This is necessary for PDF.js to be able to fetch the document.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/analyze_document", methods=["POST"])
def analyze_document():
    """
    Takes a filename, reads the full PDF text, and asks the AI to
    identify outdated sections and areas for improvement.
    """
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "Filename is required."}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found."}), 404

    print(f"Analyzing document: {filename}")
    full_text = ""
    try:
        reader = PdfReader(filepath)
        for page in reader.pages:
            full_text += page.extract_text() + "\n\n"
    except Exception as e:
        return jsonify({"error": f"Failed to read PDF: {str(e)}"}), 500

    # This is a highly specific prompt designed to get structured JSON output.
    # It's crucial for making the highlighting feature work reliably.
    PROMPT_FOR_ANALYSIS = f"""
        You are a professional policy analyst for a hospital in Illinois, USA.
        Your task is to review the following hospital policy document and identify two types of issues:
        1.  **Outdated Sections**: Find specific sentences or phrases that are factually outdated (e.g., refer to old technology like 'Internet Explorer', old dates, or deprecated procedures).
        2.  **Suggested Improvements**: Find specific sections where the policy could be improved, clarified, or modernized (e.g., suggesting a digital ticketing system instead of phone calls).

        **CRITICAL**: You MUST return your analysis as a single, valid JSON object. Do not include any text or markdown formatting before or after the JSON object.

        The JSON object must have two keys: "outdated" and "improvements".
        Each key should contain a list of objects.
        Each object in the list must have two keys:
        - "quote": The exact, verbatim text from the document that is outdated or needs improvement. This must be a direct copy-paste from the text.
        - "explanation": A brief explanation of why it's outdated or how it could be improved.

        Here is an example of the required JSON format:
        {{
          "outdated": [
            {{
              "quote": "All employees must use Internet Explorer for web browsing.",
              "explanation": "Internet Explorer is a deprecated browser with security vulnerabilities and should be replaced with a modern browser standard."
            }}
          ],
          "improvements": [
            {{
              "quote": "Employees may report issues via phone call to the IT department.",
              "explanation": "Consider implementing a modern ticketing system (e.g., Jira, Zendesk) for better tracking and resolution of IT issues."
            }}
          ]
        }}

        Now, analyze the following document text:
        ---
        {full_text}
        ---
    """

    try:
        # Using the Gemini model to get the analysis
        response = model.generate_content(PROMPT_FOR_ANALYSIS)

        # Clean the response text to ensure it's valid JSON
        # The model sometimes wraps the JSON in ```json ... ```
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()

        # Parse the JSON string into a Python dictionary
        analysis_result = json.loads(cleaned_response)

        return jsonify(analysis_result)

    except json.JSONDecodeError:
        print("\n--- ERROR: Failed to decode JSON from AI response ---")
        print(f"Raw response was:\n{response.text}\n")
        return jsonify({"error": "AI returned an invalid analysis format. Please try again."}), 500
    except Exception as e:
        print(f"An error occurred during AI analysis: {e}")
        return jsonify({"error": "An unexpected error occurred with the AI model."}), 500

if __name__ == "__main__":
    build_or_load_index()
    app.run(host="0.0.0.0", port=5000, debug=True)