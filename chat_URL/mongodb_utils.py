from pymongo import MongoClient, errors

def initialize_mongo_connection():
    try:
        client = MongoClient("mongodb://dpm1.adaptai.com:27017/")
        db = client['chat_app']
        print("MongoDB connection successful.")
        return db
    except errors.ConnectionError as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def save_scraped_content_to_mongo(url, cleaned_content, scraped_content_collection):
    doc_splits = text_splitter.split_documents([Document(page_content=cleaned_content)])
    text_chunks = [doc.page_content for doc in doc_splits]
    
    for chunk in text_chunks:
        scraped_content = {
            "datetime": datetime.now(),
            "url": url,
            "chunk": chunk
        }
        try:
            result = scraped_content_collection.insert_one(scraped_content)
            print(f"Scraped content chunk saved to MongoDB with ID: {result.inserted_id}")
        except Exception as e:
            print(f"Error saving content chunk to MongoDB: {e}")

def retrieve_content_by_url(url, scraped_content_collection):
    try:
        documents = scraped_content_collection.find({"url": url})
        document_count = scraped_content_collection.count_documents({"url": url})
        if document_count > 0:
            retrieved_content = []
            for document in documents:
                retrieved_content.append(document['chunk'])
            return " ".join(retrieved_content)    
        else:
            print("No documents found for the provided URL.")
            return ""
    except Exception as e:
        print(f"Error retrieving documents by URL: {e}")
        return ""
