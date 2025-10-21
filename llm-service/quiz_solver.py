from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from utils.prompt_templates import QUIZ_GRADING_PROMPT
import os
from dotenv import load_dotenv
import json

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

def grade_quiz(answer: str, user_answer: str):
    """
    정답(answer)과 사용자가 푼 답안(user_answer)을 비교하여 채점
    """
    prompt = QUIZ_GRADING_PROMPT.format(answer=answer, user_answer=user_answer)
    response = llm([HumanMessage(content=prompt)])
    
    try:
        grade_json = json.loads(response.content)
    except Exception as e:
        raise ValueError(f"채점 실패: {e}")
    
    return grade_json
