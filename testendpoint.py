import requests

def test_endpoint():
    url = 'https://b1e92cb0-8874-43a9-9d4a-e8fab6be89ed-prod.e1-us-east-azure.choreoapis.dev/g2inaction/default/v1'
    headers = {
        'accept': '*/*',
        'Authorization': 'Bearer eyJ4NXQiOiJPZDc2V3BiNGk2X1Q1dFRrWnUtUUhHX2RrNkkiLCJraWQiOiJNekF3TURObFl6YzRZemxqT0dNd00yVmxZV1kyT1RJNE56WmhNR0prT0RCa1pHWmpaR1pqTnpGa05tSXpNVEV6WkRBeU9EY3hOakZpWmpSaU9UVTVNUV9SUzI1NiIsInR5cCI6ImF0K2p3dCIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiI5NWQwMjlhOS0zNjkyLTQ5ZDUtOWIzMC01ZjBjMWE2ZTkxMDkiLCJhdXQiOiJBUFBMSUNBVElPTiIsImF1ZCI6WyJLV2gyQjNrenZhZE9rOFNqeW5wRDhCQTVxZ29hIiwiY2hvcmVvOmRlcGxveW1lbnQ6cHJvZHVjdGlvbiJdLCJuYmYiOjE3MTk1NzAwNzQsImF6cCI6IktXaDJCM2t6dmFkT2s4U2p5bnBEOEJBNXFnb2EiLCJvcmdfaWQiOiJiMWU5MmNiMC04ODc0LTQzYTktOWQ0YS1lOGZhYjZiZTg5ZWQiLCJpc3MiOiJodHRwczpcL1wvYXBpLmFzZ2FyZGVvLmlvXC90XC9hZGFwdGFpXC9vYXV0aDJcL3Rva2VuIiwiZXhwIjoxNzUxMTA2MDc0LCJvcmdfbmFtZSI6ImFkYXB0YWkiLCJpYXQiOjE3MTk1NzAwNzQsImp0aSI6ImUwMjE5OGMxLTlkNWUtNDJlYy04MTEyLTdjN2VjY2I5NmQ1YyIsImNsaWVudF9pZCI6IktXaDJCM2t6dmFkT2s4U2p5bnBEOEJBNXFnb2EifQ.c9b3w3RtkeEx5lAJLbqQ_VKsr4-aGPROIHOnJ1K7VAUKruI_nttTb72dTc5OsR9ecfCC9BQ7yQSE8XMoJNZRCVrZNonyOXTGQ2D4W4NlPVrKst3zl7RMjSTueocxks_En_zjcSAtGRvj6Fmsky9sX5hEOGx8Hx7UvuCrZk5QhdIU_h4Xj3c59GlKswGuNZIl6CX5XKwfRnSKDUmms08x4o8CgOCrVnipM7G8f4G2rXYxIPq_R3msQFl08kCvSzn6zHLVtFKpvi1kQU-NqCXqwKRxa_EhPHTfytNdGKpeE8xm866uyn8taklrhR4xKgLcZt1_495jffUiCLj7hAfLDA'
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("Response Data:", data)
            models = [model["model"] for model in data.get("models", [])]
            return models
        else:
            print("Error: Could not retrieve models - HTTP Status " + str(response.status_code))
            return None
    except requests.exceptions.RequestException as e:
        print("Error: Request failed - " + str(e))
        return None

# Test the function
models = test_endpoint()
if models:
    print("Models:", models)
else:
    print("No models retrieved.")