from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app, session, send_file, jsonify
from flask_login import login_required, current_user, logout_user
from .models import FileUpload, FileData, Question
from .chatbot_utils import find_closest_question, generate_response_from_summary, csv_to_json, truncate_text
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from . import db
import os
import msal
import time
import base64
import io
import re
import matplotlib.pyplot as plt
import matplotlib
from werkzeug.utils import secure_filename
import gradio as gr
import numpy as np
from PIL import Image
import logging
import uuid
import matplotlib.pyplot as plt
import io
from PIL import Image
from langchain.prompts import PromptTemplate
import json
import uuid
from dotenv import load_dotenv
from lida import TextGenerationConfig
import csv
import sys
import httpx
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from flask import send_from_directory
import requests
from functools import wraps
import logging

# Configure logging to display messages in the console
logging.basicConfig(level=logging.DEBUG)

#Load environment variables from .env
load_dotenv(override=True)

# Use 'Agg' backend for Matplotlib in non-GUI environments
plt.switch_backend('Agg')

views = Blueprint('views', __name__)

textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo", use_cache=True)

# Define MSAL configurations
CLIENT_ID = os.getenv('CLIENT_ID')  # Application (client) ID
CLIENT_SECRET = os.getenv('CLIENT_SECRET')  # client secret 
TENANT_ID = os.getenv('TENANT_ID')  # Directory (tenant) ID
AUTHORITY = os.getenv('AUTHORITY', 'https://login.microsoftonline.com/common')   # Authority URL
REDIRECT_URI = os.getenv('REDIRECT_URI', 'http://localhost:5000/redirect')  # Ensure this matches its registered redirect URI
SCOPE = ["User.Read"]  # Scope for MS Graph API

# OpenAI API Key loaded from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Ensure required variables are available
if not all([CLIENT_ID, CLIENT_SECRET, TENANT_ID, AUTHORITY, REDIRECT_URI, OPENAI_API_KEY]):
    raise ValueError("One or more required environment variables are missing. Please check your .env file.")

# Create an MSAL client instance
msal_client = msal.ConfidentialClientApplication(
    CLIENT_ID, 
    authority=AUTHORITY, 
    client_credential=CLIENT_SECRET
)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'txt'}

def allowed_file(filename):
    """Checking if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_large_file(filepath):
    """Process large CSV or Excel files in chunks to avoid memory overload."""
    if filepath.endswith('.csv'):
        chunks = pd.read_csv(filepath, chunksize=10000)
        return pd.concat(chunks, ignore_index=True)
    elif filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)

def oauth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        access_token = session.get('access_token')
        logging.debug(f"Access Token in Session: {access_token}")

        if 'access_token' not in session or is_token_expired():
            logging.warning("Access token missing or expired.")
            flash('You must be logged in to access this page.', category='warning')
            return redirect(url_for('views.login'))
        
        logging.debug("Access token is valid. Proceeding with the request.")
        return f(*args, **kwargs)
    return decorated_function

def is_token_expired():
    """Check if the access token is expired."""
    if 'expires_at' in session:
        return time.time() > session['expires_at']
    return True

@views.route('/')
def home():
    if 'access_token' in session and not is_token_expired():
        return render_template("home.html", user=current_user)
        import pdb; pdb.set_trace()
    else:
        flash('Your session has expired. Please log in again.', category='warning')
        return redirect(url_for('views.login'))

@views.route('/auth/callback')
def handle_auth_callback():
    """Handle the redirect from Microsoft and acquire an access token."""
    code = request.args.get('code')  # Get the authorization code
    if not code:
        flash('No code provided', category='error')
        return redirect(url_for('views.home'))
    
    # Exchange the authorization code for an access token
    token_response = msal_client.acquire_token_by_authorization_code(
        code,
        scopes=SCOPE,
        redirect_uri=REDIRECT_URI  # Ensure this matches the .env value
    )

    if 'access_token' in token_response:
        access_token = token_response['access_token']
        session['access_token'] = access_token
        session['expires_at'] = time.time() + token_response['expires_in']

        # Fetch user information from Microsoft Graph API
        graph_api_url = 'https://graph.microsoft.com/v1.0/me'
        headers = {'Authorization': f'Bearer {access_token}'}

        try:
            response = requests.get(graph_api_url, headers=headers)
            response.raise_for_status()
            user_info = response.json()

            # Extract relevant fields
            user_data = {
                "username": user_info.get('displayName', 'Unknown'),
                "email": user_info.get('mail', user_info.get('userPrincipalName', 'Unknown'))
            }

            session['user_info'] = user_data # Store user data in the session

            flash('Login successful!', category='success')
            return redirect(url_for('views.home'))  # Redirect to the home page

        except requests.exceptions.RequestException as e:
            flash(f'Failed to fetch user info: {str(e)}', category='error')
            return redirect(url_for('views.home'))
    else:
        error_message = token_response.get('error_description', 'Unknown error occurred')
        flash(f'Login failed: {error_message}', category='error')
        return redirect(url_for('views.home'))
    

@views.route('/graph')
@oauth_required
def get_graph_data():
    access_token = session.get('access_token')

    if not access_token:
        return jsonify({"error": "No access token, please log in first."}), 401

    import requests
    graph_api_url = 'https://graph.microsoft.com/v1.0/me'
    headers = {'Authorization': f'Bearer {access_token}'}
    
    response = requests.get(graph_api_url, headers=headers)
    return jsonify(response.json())
 


def classify_query(user_query):
    """Classify the query using LIDA's natural language capabilities."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"Classify this query as 'visualization' or 'general': {user_query}"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        classification = response.choices[0].message.content.strip().lower()
        return classification == "visualization"
    except Exception as e:
        print(f"Error in classification: {e}")
        return False

def csv_to_json(file_path):
    """Convert a CSV file to a JSON structure."""
    try:
        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            columns = reader.fieldnames
            
            if not columns:
                raise ValueError("The CSV file is empty or has no headers.")
            
            data = {
                "file_name": os.path.splitext(os.path.basename(file_path))[0],
                "columns": [{col: []} for col in columns]
            }
            
            for row in reader:
                for idx, col in enumerate(columns):
                    data["columns"][idx][col].append(row[col])
            
            return data
    except Exception as e:
        print(f"Error converting CSV to JSON: {e}")
        return None
@views.route('/debug-session')
def debug_session():
    return jsonify(dict(session))

@views.route('/upload', methods=['GET', 'POST'])
@oauth_required
def upload():
    """Render the upload page on GET and handle file uploads on POST."""
    if request.method == 'GET':
        # Render the upload.html template
        return render_template('upload.html')

    # POST request: Handle file upload
    user_info = session.get('user_info', {})
    username = user_info.get('preferred_username', 'Unknown')

    upload_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], username)
    os.makedirs(upload_folder, exist_ok=True)

    lida = current_app.config.get('LIDA_MANAGER')
    if not lida:
        return jsonify({"error": "LIDA manager is not configured. Please contact support."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file selected. Please choose a file to upload."}), 400

    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload a valid CSV or Excel file."}), 400

    # Secure the filename and handle duplicates
    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    if os.path.exists(file_path):
        filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(upload_folder, filename)

    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Error saving file '{filename}': {str(e)}"}), 500

    try:
        summary = lida.summarize(file_path, summary_method="default", textgen_config=textgen_config)
        file_data_json = csv_to_json(file_path) if filename.endswith('.csv') else None

        file_data = FileData(
            filename=filename,
            file_path=file_path,
            file_data=file_data_json,
            summary=summary,
            username=username
        )
        db.session.add(file_data)
        db.session.commit()

        goals = lida.goals(summary=summary, n=6, textgen_config=textgen_config)
        for goal in goals:
            db.session.add(Question(question_text=goal.question.strip(), file_data_id=file_data.id))

        db.session.commit()
        session['file_path'] = file_path
        session['filename'] = filename

        return jsonify({"message": f"File '{filename}' uploaded and processed successfully."}), 200

    except Exception as e:
        return jsonify({"error": f"Error processing file '{filename}': {str(e)}"}), 500

@views.route('/generated_plots/<filename>')
def get_generated_plot(filename):
    """Serve the generated plot image from the generated_plots directory."""
    image_folder = os.path.join(current_app.root_path, 'generated_plots')
    return send_from_directory(image_folder, filename)

def process_query(user_query, file_data):
    """Process the user query using LangChain Agent and dynamically generate a plot if requested."""
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    agent = create_pandas_dataframe_agent(llm, file_data, verbose=True, allow_dangerous_code=True)

    # Generate a unique filename for each plot image
    unique_filename = f"generated_image_{uuid.uuid4()}.png"
    image_folder = os.path.abspath("C:/Users/ceren/OneDrive/Masaüstü/Resume/website-flask/generated_plots")
    image_path = os.path.join(image_folder, unique_filename)

    # Ensure the folder exists
    os.makedirs(image_folder, exist_ok=True)

    prompt_template = PromptTemplate(
        input_variables=["user_query"],
        template=(
            """Query: {user_query}
            If a plot is generated, follow these steps:
            1. Save it as a PNG image at image_path = "{image_path}"
            2. Once saved, close/clear the plot.
            Only respond with JSON format. Output format: {{"text": <result_text>, "image": <image_path or null>}}."""
        )
    )

    prompt = prompt_template.format(user_query=user_query, image_path=image_path.replace("\\", "/"))

    try:
        response = agent.invoke(prompt)
        current_app.logger.debug(f"Raw Response: {response}")

        if isinstance(response, dict) and 'output' in response:
            response_output = json.loads(response['output'].replace('None', 'null'))
            response_text = response_output.get("text", "Sorry, I couldn't understand that.")
            image_path_in_response = response_output.get("image", None)

            if image_path_in_response and os.path.exists(image_path_in_response):
                session['image_path'] = image_path_in_response  # Save image path in session
                with open(image_path_in_response, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                return response_text, img_base64

    except json.JSONDecodeError as e:
        current_app.logger.error(f"JSON decode error: {e}")
        return "Error parsing response JSON.", None
    except Exception as e:
        current_app.logger.error(f"Error during agent invocation: {e}")
        return "Sorry, an error occurred while processing the query.", None

    return "Sorry, no response was generated.", None

@views.route('/execute_query', methods=['POST'])
def execute_query():
    """Execute the user query through LangChain and generate a plot if requested."""
    user_query = request.json.get("query", "")
    file_path = session.get('file_path')

    if not file_path:
        return jsonify({"error": "No file uploaded."})

    file_data = process_large_file(file_path)  # Load data    

    # Process the query using the LangChain agent
    response_text, image_path = process_query(user_query, file_data)
    current_app.logger.debug(f"Response Text: {response_text}")
    current_app.logger.debug(f"Image URL: {image_path}")

    return jsonify({"result": response_text, "image": image_path})  # Return result and image path

def process_general_query(user_query):
    """Generate a response for general queries using LIDA logic (OpenAI ChatCompletion)."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    messages = [
        {"role": "system", "content": "You are an intelligent assistant."},
        {"role": "user", "content": user_query}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
    
def route_query(user_query, file_data=None):
    if file_data is not None and classify_query(user_query):
        response = process_query(user_query, file_data)
        if response is None:
            return "Error processing the query.", None
        return response
    else:
        return process_general_query(user_query), None
    
@views.route('/visualize', methods=['GET', 'POST'])
@oauth_required
def visualize():
    """Handle both LIDA and LangChain Agent operations dynamically."""
    user_info = session.get('user_info', {})
    username = user_info.get('preferred_username', 'Unknown')

    if 'file_path' not in session:
        flash('No file uploaded. Please upload a file first.', category='warning')
        return redirect(url_for('views.upload'))

    file_path = session['file_path']
    file_data = process_large_file(file_path)

    # Check if file data is valid
    if file_data is None:
        return jsonify({"error": "Invalid file data. Please upload a valid file."})

    if request.method == 'POST':
        user_message = request.json.get('message', '').strip()

        if not user_message:
            return jsonify({"error": "Please provide a valid query."})

        current_app.logger.debug(f"User query: {user_message}")
        current_app.logger.debug(f"File path: {file_path}")

        # Classify the query type using LIDA
        if classify_query(user_message):
            response_text, image_base64 = process_query(user_message, file_data)

            if response_text is None:
                response_text = "Sorry, I couldn't process the query."

            if image_base64:
                return jsonify({"response": response_text, "image": image_base64})
            else:
                return jsonify({"response": response_text})

        else:
            # Handle general queries using LIDA
            user_files = FileData.query.filter_by(username=username).all()
            summaries = [file.summary for file in user_files if file.summary]
            combined_summary = "\n\n".join([str(s) for s in summaries])

            try:
                truncated_summary = truncate_text(combined_summary, max_tokens=10000)
                response_text = generate_response_from_summary(truncated_summary, user_message)
                return jsonify({"response": response_text})
            except Exception as e:
                current_app.logger.error(f"Error generating response: {e}")
                return jsonify({"error": "Error generating response."})

    return render_template('visualize.html')

@views.route('/download_image', methods=['POST'])
def download_image():
    format = request.form.get('format')

    # Ensure image path is in session
    if 'image_path' not in session:
        flash("Image not found. Please generate an image first.")
        logging.debug("Download attempt without an image in the session.")
        return redirect(url_for('views.visualize'))

    try:
        # Load the image from the saved path
        image_path = session['image_path']
        image = Image.open(image_path)

        # Create a buffer to store the output file
        output = io.BytesIO()

        if format == 'png':
            # Save the image as PNG to the buffer
            image.save(output, format='PNG')
            output.seek(0)
            return send_file(
                output,
                mimetype='image/png',
                as_attachment=True,
                download_name='image.png'
            )
        elif format == 'svg':
            # Save the image as SVG using matplotlib
            fig, ax = plt.subplots()
            ax.axis('off')  # Turn off axis
            # Convert the image to a numpy array for matplotlib
            ax.imshow(np.array(image))
            fig.savefig(output, format='svg', bbox_inches='tight')
            plt.close(fig)  # Close the plot to free memory
            output.seek(0)
            return send_file(
                output,
                mimetype='image/svg+xml',
                as_attachment=True,
                download_name='image.svg'
            )
        elif format == 'pdf':
            # Save the image as a PDF to the buffer
            image.convert('RGB').save(output, format='PDF', resolution=100)
            output.seek(0)
            return send_file(
                output,
                mimetype='application/pdf',
                as_attachment=True,
                download_name='image.pdf'
            )
        else:
            # Handle unsupported formats
            logging.error(f"Unsupported format requested: {format}")
            return "Invalid format. Please select png, svg, or pdf.", 400
    except Exception as e:
        # Log the error for debugging
        logging.error(f"Error while generating the download: {e}")
        flash("An error occurred while processing the download request.")
        return redirect(url_for('views.visualize'))

from flask import Blueprint, redirect, url_for, session, request, jsonify
from flask import current_app as app
import uuid

@views.route('/login')
def login():
    # Generate the authorization URL
    auth_url = current_app.config['MSAL_CLIENT'].get_authorization_request_url(
        scopes=["User.Read"],
        state=str(uuid.uuid4()),
        redirect_uri=current_app.config['REDIRECT_URI']
    )
    return redirect(auth_url)

@views.route('/logout')
def logout():
    logout_user()  # Log out the user
    session.clear()  # Clear session
    flash('You have been logged out.', category='info')
    return redirect(url_for('views.home'))
