import os
import numpy as np
import faiss
import pickle
# UPDATED IMPORTS: Added session, redirect, url_for, flash, and wraps
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
from dotenv import load_dotenv
from pypdf import PdfReader
# from google import genai
from werkzeug.utils import secure_filename
import uuid
from flask import Response, stream_with_context, send_from_directory
import time
import json
import google.generativeai as genai
from functools import wraps  # Added for login decorator

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

# --- NEW: SECRET KEY FOR SESSION MANAGEMENT ---
# A secret key is required for Flask sessions to work.
# In production, use a long, random string and store it securely.
app.config['SECRET_KEY'] = 'your-super-secret-key-that-you-should-change'

# --- NEW: HARDCODED USER CREDENTIALS ---
# For this simple example, we'll hardcode the login credentials.
# Replace these with your desired username and password.
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

FAISS_INDEX_PATH = "faiss_index.idx"
METADATA_PATH = "metadata.pkl"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# FAISS_INDEX = None
# DOC_CHUNKS = []
# METADATA = []

AI_MODEL = "gemini-2.5-flash"

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(AI_MODEL)
except Exception as e:
    print(f"Error configuring GenerativeAI: {e}")
    # Handle the error appropriately, maybe exit or use a fallback
    model = None

SYSTEM_PROMPT = """
You are a specialized AI assistant for a hospital.
Your primary function is to answer questions based on the hospital's internal policy documents.
However, you can also engage in general conversation.

Organization Information (Context):
- The organization is a hospital located in Illinois, USA with Over 4000 employees.
- All advice, interpretations, and suggestions must strictly adhere to HIPAA regulations and general healthcare compliance standards.
- When answering from documents and be precise.
- When engaging in general conversation, be helpful and polite.
"""


# --- NEW: LOGIN REQUIRED DECORATOR ---
def login_required(f):
    """
    A decorator to ensure a user is logged in before accessing a route.
    Redirects to the login page if the user is not authenticated.
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)

    return decorated_function


# REPLACE your old function with this one

def build_or_load_index(force_rebuild=False):
    """
    MODIFIED: Now loads from disk and RETURNS the index and metadata.
    Does not use global variables.
    """
    print("\n--- Running build_or_load_index (Stateless) ---")

    if force_rebuild and os.path.exists(FAISS_INDEX_PATH):
        print("Forcing rebuild: Deleting old index files.")
        if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
        if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)

    # If files exist and we are NOT forcing a rebuild, load and return them.
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH) and not force_rebuild:
        print(f"Loading existing index from {FAISS_INDEX_PATH}...")
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, "rb") as f:
                data = pickle.load(f)
                doc_chunks = data["chunks"]
                metadata = data["metadata"]
            print(f"Index loaded successfully. Total vectors: {faiss_index.ntotal}")
            return faiss_index, doc_chunks, metadata
        except Exception as e:
            print(f"Error loading index files: {e}. Rebuilding...")
            # If loading fails, proceed to rebuild.

    # --- Rebuild Logic (largely the same) ---
    print("Building a new index...")
    all_chunks = []
    all_metadata = []

    doc_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    print(f"Found {len(doc_files)} documents to index: {doc_files}")

    if not doc_files:
        print("No documents found in the upload folder. Returning empty index.")
        return None, [], []

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
            # ... (your .txt file logic can remain here if you have it)
        except Exception as e:
            print(f"   [ERROR] Failed to process {doc_file}: {e}")

    if not all_chunks:
        print("No valid text chunks were extracted. Returning empty index.")
        return None, [], []

    print(f"Generated {len(all_chunks)} total chunks. Now generating embeddings...")
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=all_chunks,
        task_type="RETRIEVAL_DOCUMENT"
    )

    embeddings = response['embedding']
    embedding_matrix = np.array(embeddings, dtype='float32')

    embedding_dimension = embedding_matrix.shape[1]
    new_faiss_index = faiss.IndexFlatL2(embedding_dimension)
    new_faiss_index.add(embedding_matrix)

    print(f"Saving index and metadata to {FAISS_INDEX_PATH}...")
    faiss.write_index(new_faiss_index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": all_metadata}, f)

    print(f"Index built and saved successfully. Total vectors: {new_faiss_index.ntotal}")
    return new_faiss_index, all_chunks, all_metadata


# REPLACE your old function with this one

def answer_query_stream(question, selected_docs, k=8):
    """
    MODIFIED: Now loads the index at the start of every request to be stateless.
    """
    import json

    # --- ADD THIS BLOCK AT THE VERY BEGINNING ---
    # Load the latest index and metadata from disk every time.
    try:
        faiss_index, doc_chunks, metadata = build_or_load_index()
    except Exception as e:
        print(f"FATAL: Could not load index on the fly: {e}")
        faiss_index, doc_chunks, metadata = None, [], []
    # --- END OF ADDED BLOCK ---


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
        metadata_payload = {"sources": [], "follow_up": ANALYTICAL_QUESTIONS}
        yield json.dumps({"type": "metadata", "data": metadata_payload}) + '\n'
        return

    # --- UPDATED LOGIC ---
    # Check the locally loaded faiss_index, not the global one.
    if faiss_index is None or faiss_index.ntotal == 0:
        error_message = "The document index is empty. Please upload documents first."
        yield json.dumps({"type": "token", "data": error_message}) + '\n'
        metadata_payload = {"sources": [], "follow_up": []}
        yield json.dumps({"type": "metadata", "data": metadata_payload}) + '\n'
        return

    response = genai.embed_content(
        model="models/text-embedding-004",
        content=question,
        task_type="RETRIEVAL_QUERY"
    )

    query_embedding = np.array([response['embedding']], dtype='float32')

    # Use the locally loaded faiss_index
    distances, indices = faiss_index.search(query_embedding, faiss_index.ntotal)

    filtered_results = []
    distance_threshold = 1.5 if question in ANALYTICAL_QUESTIONS else 1.2
    for i, dist in zip(indices[0], distances[0]):
        # Use the locally loaded metadata
        if metadata[i]['source'] in selected_docs and dist < distance_threshold:
            filtered_results.append({'index': i})

    prompt_to_use = ""
    sources = []

    if not filtered_results:
        prompt_to_use = f"{SYSTEM_PROMPT}\n\nThe user asked: '{question}'. There are no relevant policy documents found. Answer as a general AI assistant."
        sources = []
    else:
        top_k_indices = [res['index'] for res in filtered_results[:k]]
        # Use locally loaded doc_chunks and metadata
        docs = [doc_chunks[i] for i in top_k_indices]
        sources = sorted(list(set([f"{metadata[i]['source']} (Page {metadata[i]['page']})" for i in top_k_indices])))
        context = "\n\n---\n\n".join(docs)

        if question in ANALYTICAL_QUESTIONS:
            print("Using ANALYTICAL prompt...")
            prompt_to_use = f"""{SYSTEM_PROMPT}\n\nYou are acting as an expert policy analyst... \n\nContext from policy documents:\n{context}\n\nUser's Analytical Request:\n"{question}"\n\nYour detailed analysis:"""
        else:
            print("Using standard RAG prompt...")
            prompt_to_use = f"""{SYSTEM_PROMPT}\n\nYou MUST answer the question using ONLY the provided context...\n\nContext from policy documents:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""

    print(f"\n>>> Generating content...")
    try:
        stream = model.generate_content(prompt_to_use, stream=True)
        for chunk in stream:
            if hasattr(chunk, 'text') and chunk.text:
                data = {"type": "token", "data": chunk.text}
                yield json.dumps(data) + '\n'
        print(">>> Stream finished. Yielding metadata.")
        final_metadata = {"type": "metadata", "data": {"sources": sources, "follow_up": ANALYTICAL_QUESTIONS}}
        yield json.dumps(final_metadata) + '\n'
    except Exception as e:
        print(f"\n--- ERROR DURING STREAM GENERATION: {e} ---\n")
        error_message = "An error occurred with the AI model. Please check the server logs."
        data = {"type": "token", "data": error_message}
        yield json.dumps(data) + '\n'
        final_metadata = {"type": "metadata", "data": {"sources": [], "follow_up": []}}
        yield json.dumps(final_metadata) + '\n'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- NEW: LOGIN AND LOGOUT ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Handles the login process.
    GET: Displays the login page.
    POST: Validates credentials and logs the user in.
    """
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            flash('You were successfully logged in!', 'success')
            # Redirect to the main chat page after login
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    """
    Logs the user out by clearing the session.
    """
    session.pop('logged_in', None)
    flash('You were logged out.', 'info')
    return redirect(url_for('login'))


# --- PROTECTED FLASK ROUTES ---
# The @login_required decorator is added to each route
# that should only be accessible after a successful login.

@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/chat")
@login_required
def chat():
    return render_template("chat.html")


@app.route("/viewer")
@login_required
def viewer_page():
    return render_template("viewer.html")


@app.route("/documents", methods=["GET"])
@login_required
def get_documents():
    docs = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    return jsonify(sorted(docs))


@app.route("/upload", methods=["POST"])
@login_required
def upload_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type or no file selected"}), 400

    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    build_or_load_index(force_rebuild=True)

    return jsonify({"success": f"'{filename}' uploaded successfully."})


@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/ask", methods=["POST"])
@login_required
def ask():
    data = request.get_json(force=True)
    q = data.get("query")
    selected_docs = data.get("selected_docs", [])

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


@app.route("/analyze_document", methods=["POST"])
@login_required
def analyze_document():
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "Filename is required."}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found."}), 404

    print(f"Analyzing document for summary & findings: {filename}")
    full_text = ""
    try:
        reader = PdfReader(filepath)
        text_limit = 30000
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"
            if len(full_text) > text_limit:
                break
    except Exception as e:
        return jsonify({"error": f"Failed to read PDF: {str(e)}"}), 500

    PROMPT_FOR_ANALYSIS = f"""
        You are a professional policy analyst for a hospital in Illinois, USA.
        Your task is to review the following hospital policy document and provide a complete analysis.

        **CRITICAL**: You MUST return your analysis as a single, valid JSON object.
        Do not include any text, notes, or markdown formatting like ```json before or after the JSON object.

        The JSON object must have three top-level keys:
        1.  "summary": A concise 2-3 paragraph summary of the document's purpose, scope, and key points.
        2.  "outdated": A list of objects, where each object identifies a specific outdated section.
        3.  "improvements": A list of objects, where each object identifies a section that could be improved.

        For the "outdated" and "improvements" lists, each object must have two keys:
        - "quote": The exact, verbatim text from the document. This must be a direct copy-paste.
        - "explanation": A brief explanation of the issue or suggestion.

        Here is an example of the required JSON format:
        {{
          "summary": "This document outlines the hospital's policy regarding the use of information technology resources...",
          "outdated": [
            {{
              "quote": "All employees must use Internet Explorer for web browsing.",
              "explanation": "Internet Explorer is a deprecated browser with security vulnerabilities..."
            }}
          ],
          "improvements": [
            {{
              "quote": "Employees may report issues via phone call to the IT department.",
              "explanation": "Consider implementing a modern ticketing system..."
            }}
          ]
        }}

        Now, analyze the following document text:
        ---
        {full_text}
        ---
    """

    try:
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
        model_for_analysis = genai.GenerativeModel(AI_MODEL, generation_config=generation_config)

        response = model_for_analysis.generate_content(PROMPT_FOR_ANALYSIS)
        analysis_result = json.loads(response.text)
        return jsonify(analysis_result)

    except (json.JSONDecodeError, ValueError) as e:
        print(f"\n--- ERROR: Failed to decode JSON from AI response. Error: {e} ---")
        print(f"Raw response was:\n{getattr(response, 'text', 'No text in response')}\n")
        return jsonify({"error": "AI returned an invalid analysis format. Please try again."}), 500
    except Exception as e:
        print(f"An error occurred during AI analysis: {e}")
        return jsonify({"error": "An unexpected error occurred with the AI model."}), 500


if __name__ == "__main__":
    build_or_load_index()
    app.run(host="0.0.0.0", port=8000, debug=False)