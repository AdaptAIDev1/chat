import requests

def send_prompt_to_api(prompt, model_name):
    url = 'https://b1e92cb0-8874-43a9-9d4a-e8fab6be89ed-prod.e1-us-east-azure.choreoapis.dev/g2inaction/processchat/v1'
    headers = {
        'accept': '*/*',
        'Content-Type': 'application/json',
        'Authorization': 'Bearer eyJ4NXQiOiJPZDc2V3BiNGk2X1Q1dFRrWnUtUUhHX2RrNkkiLCJraWQiOiJNekF3TURObFl6YzY0WXpsak9HTXdNMlZsWldZMk9USTROelpoTUdKcU9EQmtZeldqWkdakE56RmtObUl6TVRFeloRAAyOEAAsDfxNZXFI2ISpYHz8xBCpmQOIE9lIETdCS9rM2o1dIidHSTEuLyOk2bcxkAKdmGHQADbglskRaIQ8DafM8CWQWsmEXgS2DYgR1EHsmcRQDUhEX5Fi'
    }
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            if 'response' in response_data:
                return response_data['response'], model_name
            else:
                return "Response key is missing in the API response.", model_name
        else:
            error_message = f"Error: {response.status_code} - {response.json().get('error', response.text)}"
            return error_message, model_name
    except requests.RequestException as e:
        return f"Error sending POST request: {e}", model_name

# Test the function
if __name__ == '__main__':
    prompt = "What is the capital of France?"
    model_name = "llama3"  # Replace with your model name
    response, model_name = send_prompt_to_api(prompt, model_name)
    print("Response:", response)
    print("Model Name:", model_name)