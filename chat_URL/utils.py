import numpy as np
import sqlite3
import requests

def store_context(chunks, embeddings):
    conn = sqlite3.connect('context.db')
    c = conn.cursor()
    # Drop the table if it exists to avoid dimension inconsistency
    c.execute('''DROP TABLE IF EXISTS context''')
    c.execute('''CREATE TABLE context (chunk TEXT, embedding BLOB)''')
    conn.commit()
    
    for chunk, embedding in zip(chunks, embeddings):
        embedding_blob = np.array(embedding).tobytes()
        c.execute("INSERT INTO context (chunk, embedding) VALUES (?, ?)", (chunk, embedding_blob))
    conn.commit()
    conn.close()

def send_prompt_to_local_llm(prompt, model_name):
    url = "http://199.204.135.71:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_text = response.text
        try:
            response_data = response.json()
            if 'response' in response_data:
                return response_data['response'], model_name
            else:
                return "Response key is missing in the API response.", model_name
        except ValueError as json_err:
            return f"Error decoding JSON response: {json_err} - Response text: {response_text}", model_name
    except requests.RequestException as e:
        return f"Error sending POST request: {e}", model_name
