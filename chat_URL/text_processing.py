import re

def clean_content(content):
    # Initial metrics
    original_length = len(content)
    original_lines = len(content.splitlines())

    content = re.sub(r'\[.*?\]\(.*?\)', '', content)
    content = re.sub(r'!\[\]\(', '', content)
    lines = content.splitlines()
    unique_lines = []
    for line in lines:
        if line.strip() and line not in unique_lines:
            unique_lines.append(line)
    cleaned_content = '\n'.join(unique_lines)
    cleaned_content = re.sub(r'\s+', ' ', cleaned_content)

    # Final metrics
    cleaned_length = len(cleaned_content)
    cleaned_lines = len(cleaned_content.splitlines())

    # Metrics dictionary
    metrics = {
        "original_length": original_length,
        "cleaned_length": cleaned_length,
        "original_lines": original_lines,
        "cleaned_lines": cleaned_lines,
    }

    return cleaned_content, metrics

class LocalEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        if not texts:
            raise ValueError("No texts provided for embedding")
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=True)[0].tolist()
