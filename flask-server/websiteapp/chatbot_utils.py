from flask import current_app
from difflib import SequenceMatcher
from .models import FileData
from tiktoken import get_encoding
import csv
import os
import logging

# Use SequenceMatcher to find the closest matching question
def find_closest_question(user_input, questions):
    best_match = None
    highest_ratio = 0
    for question in questions:
        ratio = SequenceMatcher(None, user_input, question.question_text).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = question
    return best_match

def truncate_text(text, max_tokens=10000):
    """Truncate the text to fit within the max_tokens limit."""
    try:
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")

        encoding = get_encoding("cl100k_base")  # Tokenizer for gpt-3.5-turbo
        tokens = encoding.encode(text)
        logging.debug(f"Original token count: {len(tokens)}")

        truncated_tokens = tokens[:max_tokens]
        logging.debug(f"Truncated token count: {len(truncated_tokens)}")

        return encoding.decode(truncated_tokens)

    except Exception as e:
        logging.error(f"Error truncating text: {e}")
        # Fallback to character-based truncation if tokenization fails
        return text[:max_tokens]

def generate_response_from_summary(summary, user_query):
    # Access the OpenAI client from the app config
    openai_client = current_app.config['OPENAI_CLIENT']

    # Construct a detailed prompt to guide the model
    messages = [
        {"role": "system", "content": "You are a data analysis assistant. Use the following summary of the data to answer questions accurately. If the question requires visualization, acknowledge the request and suggest generating a plot."},
        {"role": "user", "content": f"Data Summary:\n{summary}"},
        {"role": "user", "content": f"Question: {user_query}"}
    ]

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.5
        )

        # Extract and return the content of the assistant's reply
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"
    

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
    
def generate_response_from_file_data(file_id, user_query):
    """
    Generate a response using OpenAI's ChatCompletion based on file_data in the database.

    Args:
        file_id (int): The ID of the file_data in the database.
        user_query (str): The user's query.

    Returns:
        str: The assistant's response.
    """
    try:
        # Fetch the file_data from the database
        file_data_entry = FileData.query.get(file_id)

        if not file_data_entry or not file_data_entry.file_data:
            return "Error: The specified file data is not available or does not contain data."

        # Extract file_data
        file_data = file_data_entry.file_data

        # Access the OpenAI client from the app config
        openai_client = current_app.config['OPENAI_CLIENT']

        # Construct a prompt in chat format
        messages = [
            {"role": "system", "content": "You are an intelligent assistant. Use the following data to answer questions."},
            {"role": "user", "content": f"Data: {file_data}"},
            {"role": "user", "content": f"Question: {user_query}"}
        ]

        # Generate response from OpenAI using ChatCompletion
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.5
        )

        # Extract and return the content of the assistant's reply
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"