import os
import json
import logging
import subprocess
from flask import Flask, request, jsonify, render_template, abort, send_from_directory
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from constants import OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

CONFIG_FILE = 'config.json'
CHAT_HISTORY_FILE = 'chat_history.json'
UPLOAD_FOLDER = 'pdf'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_config():
    """Load the configuration from the JSON file."""
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def save_config(config):
    """Save the configuration to the JSON file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def load_chat_history():
    """Load chat history from file."""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as file:
            return json.load(file)
    return []

def save_chat_history(history):
    """Save chat history to file."""
    with open(CHAT_HISTORY_FILE, 'w') as file:
        json.dump(history, file, indent=4)

def fetch_available_models():
    """Fetch available Ollama models."""
    try:
        result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        output = result.stdout
        logging.debug(f"Raw output from `ollama list`:\n{output}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running ollama list: {e}")
        return []

    models = []
    excluded_keywords = {'tge', 'bge', 'text', 'embed'}
    for line in output.splitlines():
        if line.strip() and not line.startswith("NAME"):
            parts = line.split()
            if parts:
                model_name = parts[0].strip()
                if not any(keyword in model_name for keyword in excluded_keywords):
                    models.append(model_name)
    return models

def update_config_with_models(models):
    """Update config with available models."""
    config = load_config()
    config['available_models'] = models
    save_config(config)
    logging.info("Config updated with latest models.")

# Load configuration
config = load_config()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(model=config['embeddings_model'])

# Initialize Ollama LLM
llm = Ollama(model=config['llm_model'])

# Load and process the PDF
def process_pdf(pdf_path):
    logging.info(f"Processing PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} pages from the PDF")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks")

    return chunks

# Create QA Chain
def create_qa_chain(vectorstore):
    prompt_template = """Use the following context to answer the question at the end. If the context is not sufficient to answer the question accurately, please state that the information is not available. Provide a confidence score between 0 and 1 for your answer, where 1 means high confidence and 0 means low confidence.

    {context}

    Question: {question}
    Answer: """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    logging.info("Created RetrievalQA chain")
    return qa_chain

# Find the PDF file
pdf_path = os.path.join(config['pdf_folder'], config['pdf_file'])

if not os.path.exists(pdf_path):
    logging.error(f"PDF file not found: {pdf_path}")
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

# Check if embeddings exist, if not create and save them
embeddings_path = config['embeddings_path']
if not os.path.exists(embeddings_path):
    os.makedirs(embeddings_path)

vectorstore_path = os.path.join(embeddings_path, config['vectorstore_file'])

if os.path.exists(vectorstore_path):
    logging.info("Loading existing vectorstore")
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
else:
    logging.info("Creating new vectorstore")
    chunks = process_pdf(pdf_path)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(vectorstore_path)

# Create QA chain
qa_chain = create_qa_chain(vectorstore)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')  # Render the index.html template

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    model = data.get('model')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Refresh the available models before checking
    models = fetch_available_models()
    if models:
        update_config_with_models(models)

    if model not in config.get('available_models', []):
        return jsonify({"error": "Invalid model selected"}), 400

    logging.info(f"Received question: {question} using model: {model}")

    try:
        # Update the llm model dynamically if needed
        global llm
        llm = Ollama(model=model)
        qa_chain = create_qa_chain(vectorstore)
        
        result = qa_chain({"query": question})
        answer = result['result']
        source_docs = result['source_documents']

        # Calculate confidence score
        confidence_score = calculate_confidence(source_docs)

        # Extract source details
        if source_docs:
            source_doc = source_docs[0]
            page_number = source_doc.metadata.get('page', 'Unknown')
            chunk_number = "Unknown"  # We don't have the original chunk info here
            source_text = source_doc.page_content[:500]  # Truncate for brevity
        else:
            page_number = "Unknown"
            chunk_number = "Unknown"
            source_text = "No source found"

        response = {
            "Answer": answer,
            "Confidence Score": confidence_score,
            "Page": page_number,
            "Chunk": chunk_number,
            "Source": source_text
        }

        logging.info(f"Generated response: {response}")

        # Save to chat history
        history = load_chat_history()
        history.append({
            "question": question,
            "answer": answer,
            "confidence_score": confidence_score,
            "page": page_number,
            "chunk": chunk_number,
            "source": source_text
        })
        save_chat_history(history)

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error processing question: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    models = config.get('available_models', [])
    return jsonify({"models": models})

@app.route('/history', methods=['GET'])
def get_history():
    history = load_chat_history()
    return jsonify(history)

@app.route('/set_model', methods=['POST'])
def set_model():
    """Set the last chosen model."""
    data = request.get_json()
    model = data.get('model')
    if not model:
        abort(400, description="Model is required")

    # Refresh the available models before setting
    models = fetch_available_models()
    if models:
        update_config_with_models(models)

    if model not in config.get("available_models", []):
        abort(400, description="Model not available")

    config["last_chosen_model"] = model
    save_config(config)
    return jsonify({"message": "Model updated successfully", "last_chosen_model": model})

@app.route('/get_last_model', methods=['GET'])
def get_last_model():
    """Get the last chosen model."""
    return jsonify({"last_chosen_model": config.get("last_chosen_model", "")})

@app.route('/refresh_models', methods=['POST'])
def refresh_models():
    """Refresh the list of available models."""
    models = fetch_available_models()
    if models:
        update_config_with_models(models)
        return jsonify({"message": "Models refreshed successfully", "models": models})
    else:
        return jsonify({"error": "Failed to fetch models"}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the new PDF
        chunks = process_pdf(file_path)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(vectorstore_path)
        
        return jsonify({"message": "File uploaded successfully", "filename": filename}), 200
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/pdf/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def calculate_confidence(source_docs):
    """Calculate confidence score based on the number of source documents."""
    return min(len(source_docs) / 3, 1.0)  # Adjust denominator as needed

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Fetch and update models on startup
    models = fetch_available_models()
    if models:
        update_config_with_models(models)
    app.run(host='0.0.0.0', port=5005, debug=True)
