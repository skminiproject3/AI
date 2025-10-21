import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

# .env 파일 불러오기
load_dotenv(dotenv_path=r"C:\AI활용\00. 팀프로젝트\MINI3\ai_module\.env")

# 환경변수에서 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    temperature=0,
    openai_api_key=api_key
)

response = llm.predict("안녕하세요?")
print(response)
