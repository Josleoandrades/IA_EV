from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, DEFAULT_LLM_MODEL

def get_llm(temperature: float = 0.7):
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=DEFAULT_LLM_MODEL,
        temperature=temperature,
    )
    return llm
