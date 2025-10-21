from fastapi import FastAPI

app = FastAPI(title="AI Quiz Service", version="1.0")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "AI Quiz Module is running"}

from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from quiz_generator import generate_quiz
from quiz_solver import grade_quiz

import tempfile

app = FastAPI(title="AI Quiz Service", version="1.0")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "AI Quiz Module is running"}

# --- 예상문제 생성 요청 모델 ---
class QuizRequest(BaseModel):
    content: str

@app.post("/generate_quiz")
async def api_generate_quiz(request: QuizRequest):
    quiz_json = generate_quiz(request.content)
    return {"quiz": quiz_json}

# --- 사용자 풀이 채점 요청 모델 ---
class QuizSolveRequest(BaseModel):
    answer: str
    user_answer: str

@app.post("/grade_quiz")
async def api_grade_quiz(request: QuizSolveRequest):
    result = grade_quiz(request.answer, request.user_answer)
    return {"result": result}
