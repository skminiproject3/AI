from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from utils.prompt_templates import QUIZ_GENERATION_PROMPT
import os
from dotenv import load_dotenv
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# GPT LLM 초기화
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

def generate_quiz(content: str):
    """
    학습자료 content를 기반으로 예상문제 JSON 생성
    """
    prompt = QUIZ_GENERATION_PROMPT.format(content=content)
    response = llm([HumanMessage(content=prompt)])
    
    try:
        quiz_json = json.loads(response.content)
    except Exception as e:
        raise ValueError(f"문제 생성 실패: {e}")
    
    return quiz_json
