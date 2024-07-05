import requests

def load_instructions(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def send_prompt_to_api(prompt, model_name):
    url = 'https://b1e92cb0-8874-43a9-9d4a-e8fab6be89ed-prod.e1-us-east-azure.choreoapis.dev/g2inaction/processchat/v1'
    headers = {
        'accept': '*/*',
        'Content-Type': 'application/json',
        'Authorization': 'Bearer eyJ4NXQiOiJPZDc2V3BiNGk2X1Q1dFRrWnUtUUhHX2RrNkkiLCJraWQiOiJNekF3TURObFl6YzRZemxqT0dNd00yVmxZV1kyT1RJNE56WmhNR0prT0RCa1pHWmpaR1pqTnpGa05tSXpNVEV6WkRBeU9EY3hOakZpWmpSaU9UVTVNUV9SUzI1NiIsInR5cCI6ImF0K2p3dCIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiI5NWQwMjlhOS0zNjkyLTQ5ZDUtOWIzMC01ZjBjMWE2ZTkxMDkiLCJhdXQiOiJBUFBMSUNBVElPTiIsImF1ZCI6WyJLV2gyQjNrenZhZE9rOFNqeW5wRDhCQTVxZ29hIiwiY2hvcmVvOmRlcGxveW1lbnQ6cHJvZHVjdGlvbiJdLCJuYmYiOjE3MTk1NzAwNzQsImF6cCI6IktXaDJCM2t6dmFkT2s4U2p5bnBEOEJBNXFnb2EiLCJvcmdfaWQiOiJiMWU5MmNiMC04ODc0LTQzYTktOWQ0YS1lOGZhYjZiZTg5ZWQiLCJpc3MiOiJodHRwczpcL1wvYXBpLmFzZ2FyZGVvLmlvXC90XC9hZGFwdGFpXC9vYXV0aDJcL3Rva2VuIiwiZXhwIjoxNzUxMTA2MDc0LCJvcmdfbmFtZSI6ImFkYXB0YWkiLCJpYXQiOjE3MTk1NzAwNzQsImp0aSI6ImUwMjE5OGMxLTlkNWUtNDJlYy04MTEyLTdjN2VjY2I5NmQ1YyIsImNsaWVudF9pZCI6IktXaDJCM2t6dmFkT2s4U2p5bnBEOEJBNXFnb2EifQ.c9b3w3RtkeEx5lAJLbqQ_VKsr4-aGPROIHOnJ1K7VAUKruI_nttTb72dTc5OsR9ecfCC9BQ7yQSE8XMoJNZRCVrZNonyOXTGQ2D4W4NlPVrKst3zl7RMjSTueocxks_En_zjcSAtGRvj6Fmsky9sX5hEOGx8Hx7UvuCrZk5QhdIU_h4Xj3c59GlKswGuNZIl6CX5XKwfRnSKDUmms08x4o8CgOCrVnipM7G8f4G2rXYxIPq_R3msQFl08kCvSzn6zHLVtFKpvi1kQU-NqCXqwKRxa_EhPHTfytNdGKpeE8xm866uyn8taklrhR4xKgLcZt1_495jffUiCLj7hAfLDA'
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