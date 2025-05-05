import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_CLOUD_URL")  
QDRANT_API_KEY = os.getenv("QDRANT_CLOUD_API_KEY")
COLLECTION_NAME = "university_knowledge"