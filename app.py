import os
import re
import json
import logging
import subprocess
from flask import Flask, request, jsonify, render_template, abort, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
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
            try:
                history = json.load(file)
                # If the loaded history is a dict, convert it to a list with one item
                if isinstance(history, dict):
                    return [history]
                # If it's already a list, return it as is
                elif isinstance(history, list):
                    return history
                # If it's neither a dict nor a list, return an empty list
                else:
                    logging.warning(f"Unexpected chat history format. Returning empty list.")
                    return []
            except json.JSONDecodeError:
                logging.error(f"Error decoding chat history file. Returning empty list.")
                return []
    return []  # Return an empty list if the file doesn't exist

def save_chat_history(history):
    """Save chat history to file."""
    if not isinstance(history, list):
        logging.warning(f"Attempting to save non-list chat history. Converting to list.")
        history = [history] if history else []
    with open(CHAT_HISTORY_FILE, 'w') as file:
        json.dump(history, file, indent=4)

def fetch_available_models(model_type='llm'):
    """Fetch available Ollama models."""
    try:
        result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        output = result.stdout
        logging.debug(f"Raw output from `ollama list`:\n{output}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running ollama list: {e}")
        return []

    models = []
    embedding_keywords = {'tge', 'bge', 'text', 'embed'}
    for line in output.splitlines():
        if line.strip() and not line.startswith("NAME"):
            parts = line.split()
            if parts:
                model_name = parts[0].strip()
                is_embedding_model = any(keyword in model_name.lower() for keyword in embedding_keywords)
                if model_type == 'embeddings' and is_embedding_model:
                    models.append(f"ollama:{model_name}")
                elif model_type == 'llm' and not is_embedding_model:
                    models.append(model_name)
    return models

def update_config_with_models(models, model_type='llm'):
    """Update config with available models."""
    config = load_config()
    if model_type == 'llm':
        config['available_llm_models'] = models
    elif model_type == 'embeddings':
        ollama_models = models
        # Keep the existing non-Ollama models
        existing_models = [model for model in config.get('available_embeddings_models', []) if not model.startswith('ollama:')]
        config['available_embeddings_models'] = ollama_models + existing_models
    save_config(config)
    logging.info(f"Config updated with latest {model_type} models.")

def initialize_embeddings():
    config = load_config()
    embeddings_model = config['embeddings_model']
    if embeddings_model.startswith('openai:'):
        return OpenAIEmbeddings(model=embeddings_model.split(':')[1])
    elif embeddings_model.startswith('ollama:'):
        return OllamaEmbeddings(model=embeddings_model.split(':')[1])
    else:
        raise ValueError(f"Unsupported embeddings model: {embeddings_model}")

# Load configuration
config = load_config()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize OpenAI Embeddings
embeddings = initialize_embeddings()

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

    Format your answer using Markdown for better readability. Use paragraphs, bullet points, or numbered lists where appropriate.

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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')  # Render the index.html template

# Add this function to check embedding compatibility
def check_embedding_compatibility(vectorstore_path):
    current_embeddings = initialize_embeddings()
    try:
        vectorstore = FAISS.load_local(vectorstore_path, current_embeddings, allow_dangerous_deserialization=True)
        # Perform a test query to check compatibility
        vectorstore.similarity_search("test query", k=1)
        return True
    except AssertionError:
        return False

# Update the initialize_qa_chain function
def initialize_qa_chain():
    global qa_chain, embeddings
    
    current_embedding_model = config['embeddings_model']
    embeddings_folder = f"{current_embedding_model.replace(':', '_')}_embeddings"
    vectorstore_folder = f"{os.path.splitext(config['pdf_file'])[0]}_{embeddings_folder}"
    vectorstore_path = os.path.join(config['embeddings_path'], embeddings_folder, vectorstore_folder)

    if os.path.exists(vectorstore_path):
        embeddings = initialize_embeddings()
        if check_embedding_compatibility(vectorstore_path):
            try:
                vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
                qa_chain = create_qa_chain(vectorstore)
                logging.info(f"Successfully initialized QA chain with vectorstore from {vectorstore_path}")
            except Exception as e:
                logging.error(f"Error loading vectorstore from {vectorstore_path}: {str(e)}")
                raise
        else:
            logging.error(f"Incompatible embeddings. Recreating vectorstore for {config['pdf_file']}")
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], config['pdf_file'])
            process_pdf_and_create_embeddings(pdf_path, config['pdf_file'])
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            qa_chain = create_qa_chain(vectorstore)
    else:
        logging.error(f"Vectorstore not found for the current embedding model: {vectorstore_path}")
        raise FileNotFoundError(f"Vectorstore not found at {vectorstore_path}")

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

    if model not in config.get('available_llm_models', []):
        return jsonify({"error": "Invalid model selected"}), 400

    logging.info(f"Received question: {question} using model: {model}")

    def generate():
        yield "data: " + json.dumps({"status": "processing"}) + "\n\n"

        try:
            # Update the llm model dynamically if needed
            global llm, qa_chain
            llm = Ollama(model=model)
            
            # Reinitialize the qa_chain to use the latest vectorstore
            initialize_qa_chain()

            result = qa_chain({"query": question})
            answer = result['result']
            source_docs = result['source_documents']

            # Calculate confidence score
            confidence_score = calculate_confidence(source_docs)

            # Extract source details
            if source_docs:
                source_doc = source_docs[0]
                page_number = source_doc.metadata.get('page', 'Unknown')
                source_text = source_doc.page_content[:500]  # Truncate for brevity
            else:
                page_number = "Unknown"
                source_text = "No source found"

            # Format the answer without including confidence score and page number
            formatted_answer = format_answer(answer)

            response = {
                "Answer": formatted_answer,
                "ConfidenceScore": f"{confidence_score:.2f}",
                "Page": page_number,
                "Source": source_text
            }

            yield "data: " + json.dumps(response) + "\n\n"

            # Save to chat history
            history = load_chat_history()
            history.append({
                "question": question,
                "answer": formatted_answer,
                "confidence_score": confidence_score,
                "page": page_number,
                "source": source_text
            })
            save_chat_history(history)

        except AssertionError:
            error_message = "Embedding model mismatch. Please recreate the vectorstore with the current embedding model."
            logging.error(error_message)
            yield "data: " + json.dumps({"error": error_message}) + "\n\n"
        except Exception as e:
            logging.error(f"Error processing question: {str(e)}", exc_info=True)
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
        finally:
            yield "data: " + json.dumps({"status": "complete"}) + "\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

def format_answer(answer):
    """Format the answer as HTML for better readability."""
    paragraphs = answer.split('\n\n')
    formatted_paragraphs = []
    for para in paragraphs:
        if re.match(r'^\d+\.|\*', para.strip()):
            formatted_paragraphs.append(f"<li>{para.strip()}</li>")
        else:
            formatted_para = f"<p><strong>{para.strip().split('.')[0]}.</strong> {' '.join(para.strip().split('.')[1:])}</p>"
            formatted_paragraphs.append(formatted_para)
    
    formatted_answer = "".join(formatted_paragraphs)
    return f"<div class='formatted-answer'>{formatted_answer}</div>"

def calculate_confidence(source_docs):
    """Calculate confidence score based on the number of source documents."""
    return min(len(source_docs) / 3, 1.0)  # Adjust denominator as needed



    

@app.route('/models', methods=['GET'])
def get_models():
    llm_models = fetch_available_models('llm')
    embeddings_models = config.get('available_embeddings_models', [])  # Get from config instead of fetch
    return jsonify({"llm_models": llm_models, "embeddings_models": embeddings_models})

@app.route('/set_model', methods=['POST'])
def set_model():
    """Set the last chosen model for LLM or embeddings."""
    data = request.get_json()
    model = data.get('model')
    model_type = data.get('type', 'llm')  # 'llm' or 'embeddings'
    if not model:
        abort(400, description="Model is required")

    # Refresh the available models before setting
    models = fetch_available_models(model_type)
    if models:
        update_config_with_models(models, model_type)

    config_key = 'available_llm_models' if model_type == 'llm' else 'available_embeddings_models'
    if model not in config.get(config_key, []):
        abort(400, description="Model not available")

    if model_type == 'llm':
        config["last_chosen_llm_model"] = model
        config["llm_model"] = model
    else:
        config["last_chosen_embeddings_model"] = model
        config["embeddings_model"] = model
        
        # Check if the selected PDF is already embedded with this model
        if config.get('pdf_file'):
            update_vectorstore_path()  # Update the path to the vectorstore based on the current model
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], config['pdf_file'])

            # If the vectorstore file does not exist, create it
            if not os.path.exists(os.path.join(config['vectorstore_file'], "index.faiss")):
                process_pdf_and_create_embeddings(pdf_path, config['pdf_file'])

    save_config(config)
    
    # Reinitialize embeddings if embeddings model changed
    if model_type == 'embeddings':
        global embeddings
        embeddings = initialize_embeddings()
        # Update vectorstore_file to reflect the new embeddings model
        if config['pdf_file']:
            update_vectorstore_path()
        initialize_qa_chain()  # Reinitialize the qa_chain with the new embedding model

    return jsonify({"message": f"{model_type.capitalize()} model updated successfully", f"last_chosen_{model_type}_model": model})

def update_vectorstore_path():
    current_embedding_model = config['embeddings_model']
    embeddings_folder = f"{current_embedding_model.replace(':', '_')}_embeddings"
    vectorstore_folder = f"{os.path.splitext(config['pdf_file'])[0]}_{embeddings_folder}"
    config['vectorstore_file'] = vectorstore_folder
    save_config(config)
    logging.info(f"Updated vectorstore path to: {vectorstore_folder}")

@app.route('/get_last_model', methods=['GET'])
def get_last_model():
    """Get the last chosen model for LLM and embeddings."""
    return jsonify({
        "last_chosen_llm_model": config.get("last_chosen_llm_model", ""),
        "last_chosen_embeddings_model": config.get("last_chosen_embeddings_model", "")
    })

@app.route('/refresh_models', methods=['POST'])
def refresh_models():
    """Refresh the list of available models for both LLM and embeddings."""
    llm_models = fetch_available_models('llm')
    ollama_embeddings_models = fetch_available_models('embeddings')
    
    if llm_models:
        update_config_with_models(llm_models, 'llm')
    if ollama_embeddings_models:
        update_config_with_models(ollama_embeddings_models, 'embeddings')
    
    config = load_config()
    embeddings_models = config.get('available_embeddings_models', [])
    
    return jsonify({
        "message": "Models refreshed successfully",
        "llm_models": llm_models,
        "embeddings_models": embeddings_models
    })

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected files"}), 400
    
    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(filename)
            
            # Process the PDF and create embeddings
            process_pdf_and_create_embeddings(file_path, filename)
            
            # Update the config to reflect the current PDF
            config['pdf_file'] = filename
            config['last_selected_pdf'] = filename
            save_config(config)
    
    if uploaded_files:
        return jsonify({"message": "Files uploaded successfully", "filenames": uploaded_files}), 200
    else:
        return jsonify({"error": "No valid files were uploaded"}), 400

@app.route('/pdf/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/pdf_list', methods=['GET'])
def get_pdf_list():
    pdf_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]
    return jsonify({"pdf_files": pdf_files})

@app.route('/select_pdf', methods=['POST'])
def select_pdf():
    data = request.json
    selected_pdf = data.get('pdf_file')
    
    if not selected_pdf or not selected_pdf.endswith('.pdf'):
        return jsonify({"error": "Invalid PDF file selected"}), 400

    logging.info(f"Selecting PDF: {selected_pdf}")

    config['pdf_file'] = selected_pdf
    config['last_selected_pdf'] = selected_pdf  # Save the last selected PDF
    
    # Update the vectorstore_file to reflect the current PDF and embedding model
    update_vectorstore_path()
    
    save_config(config)
    
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_pdf)
    try:
        process_pdf_and_create_embeddings(pdf_path, selected_pdf)
        initialize_qa_chain()
        return jsonify({"message": "PDF selected and processed successfully"}), 200
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_last_pdf', methods=['GET'])
def get_last_pdf():
    """Get the last selected PDF file."""
    return jsonify({"last_selected_pdf": config.get("last_selected_pdf", "")})

def calculate_confidence(source_docs):
    """Calculate confidence score based on the number of source documents."""
    return min(len(source_docs) / 3, 1.0)  # Adjust denominator as needed

def process_pdf_and_create_embeddings(file_path, filename):
    chunks = process_pdf(file_path)
    
    # Get the current embedding model
    current_embedding_model = config['embeddings_model']
    
    # Create embeddings for the current embedding model
    embeddings_folder = f"{current_embedding_model.replace(':', '_')}_embeddings"
    vectorstore_folder = f"{os.path.splitext(filename)[0]}_{embeddings_folder}"
    vectorstore_path = os.path.join(config['embeddings_path'], embeddings_folder, vectorstore_folder)
    
    if not os.path.exists(vectorstore_path):
        # Initialize embeddings for the current model
        current_embeddings = initialize_embeddings()
        vectorstore = FAISS.from_documents(chunks, current_embeddings)
        os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)
        vectorstore.save_local(vectorstore_path)
        logging.info(f"Created and saved embeddings for {filename} using {current_embedding_model}")
    else:
        logging.info(f"Embeddings already exist for {filename} using {current_embedding_model}")

    # Update the config to use the current vectorstore
    config['vectorstore_file'] = vectorstore_folder
    save_config(config)

    # Verify that the vectorstore file was created
    if not os.path.exists(os.path.join(vectorstore_path, "index.faiss")):
        logging.error(f"Failed to create vectorstore file at {vectorstore_path}")
        raise FileNotFoundError(f"Vectorstore file not found at {vectorstore_path}")

def create_embeddings_for_all_pdfs():
    pdf_folder = config['pdf_folder']
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_folder, filename)
            try:
                process_pdf_and_create_embeddings(file_path, filename)
                logging.info(f"Successfully processed and created embeddings for {filename}")
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")

@app.route('/create_embeddings', methods=['POST'])
def create_embeddings():
    try:
        create_embeddings_for_all_pdfs()
        return jsonify({"message": "Embeddings created successfully for all PDFs"}), 200
    except Exception as e:
        logging.error(f"Error creating embeddings: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    history = load_chat_history()
    return jsonify(history)

def read_existing_api_keys():
    """Read existing API keys from constants.py"""
    existing_keys = {}
    if os.path.exists('constants.py'):
        with open('constants.py', 'r') as f:
            content = f.read()
            for line in content.splitlines():
                match = re.match(r"(\w+)\s*=\s*'([^']*)'", line)
                if match:
                    key, value = match.groups()
                    existing_keys[key] = value
    return existing_keys

@app.route('/save_api_keys', methods=['POST'])
def save_api_keys():
    data = request.json
    existing_keys = read_existing_api_keys()
    
    # Update only the keys that are provided in the request
    api_keys = {
        'OPENAI_API_KEY': data.get('openai') or existing_keys.get('OPENAI_API_KEY', ''),
        'CLAUDE_API_KEY': data.get('claude') or existing_keys.get('CLAUDE_API_KEY', ''),
        'GOOGLE_API_KEY': data.get('google') or existing_keys.get('GOOGLE_API_KEY', ''),
        'AMAZON_API_KEY': data.get('amazon') or existing_keys.get('AMAZON_API_KEY', '')
    }

    try:
        with open('constants.py', 'w') as f:
            for key, value in api_keys.items():
                f.write(f"{key} = '{value}'\n")
        return jsonify({"message": "API keys saved successfully"}), 200
    except Exception as e:
        logging.error(f"Error saving API keys: {str(e)}")
        return jsonify({"error": "Failed to save API keys"}), 500

if __name__ == '__main__':
    # Ensure the upload folder and embeddings folder exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(config['embeddings_path'], exist_ok=True)
    
    # Fetch and update models on startup
    llm_models = fetch_available_models('llm')
    ollama_embeddings_models = fetch_available_models('embeddings')
    if llm_models:
        update_config_with_models(llm_models, 'llm')
    if ollama_embeddings_models:
        update_config_with_models(ollama_embeddings_models, 'embeddings')

    # Initialize the vectorstore and qa_chain
    pdf_path = os.path.join(config['pdf_folder'], config['pdf_file'])
    try:
        process_pdf_and_create_embeddings(pdf_path, config['pdf_file'])
        initialize_qa_chain()
    except Exception as e:
        logging.error(f"Error initializing application: {str(e)}")
        # You might want to exit the application here or take appropriate action

    app.run(host='0.0.0.0', port=5005, debug=True)
