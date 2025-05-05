from langchain_cohere import ChatCohere
from config.cohere_config import COHERE_API_KEY

def get_llm():
    return ChatCohere(
        model="command-r",  
        api_key=COHERE_API_KEY,
        temperature=0.3
    )