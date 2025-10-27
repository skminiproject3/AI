import os
import re
import hashlib
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_tavily import TavilySearch
import pymysql

# ==========================================================
# 🌍 환경 변수 로드 및 초기화
# ==========================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY 미설정!")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
tavily_tool = TavilySearch(api_key=TAVILY_API_KEY, max_results=3) if TAVILY_API_KEY else None

# ==========================================================
# 🧩 FastAPI 초기화 및 CORS 설정
# ==========================================================
app = FastAPI(title="PDF 학습 도우미 API (Full 통합)")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# 🗂️ 디렉토리 자동 설정 (로컬 / 배포 둘 다 지원)
# ==========================================================
if os.getenv("ENV_MODE", "local") == "prod":
    UPLOAD_DIR = "/uploads"  # Docker 환경
else:
    UPLOAD_DIR = os.path.join(os.getcwd(), "uploaded_pdfs")

VECTOR_DIR = "vector_stores"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==========================================================
# 📄 Pydantic 모델 정의
# ==========================================================
class PdfPathsRequest(BaseModel):
    pdf_paths: List[str]

class ChapterRequest(BaseModel):
    pdf_paths: Optional[List[str]] = None
    chapter_request: Optional[str] = None

class QuestionRequest(BaseModel):
    question: str
    force_web: bool
    pdf_paths: Optional[List[str]] = None

class QuizGenerationRequest(BaseModel):
    pdf_paths: List[str]
    num_questions: int = Field(5, ge=1, le=20)
    difficulty: str = Field("MEDIUM", pattern="^(EASY|MEDIUM|HARD)$")

# ==========================================================
# 🔧 PDF 처리 유틸
# ==========================================================
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return "\n".join([p.page_content for p in pages])
    except Exception as e:
        logger.error(f"PDF 텍스트 추출 실패: {pdf_path} | {e}")
        return ""

def get_or_create_vectorstore(pdf_path: str) -> Optional[FAISS]:
    normalized_path = pdf_path.replace("\\", "/")
    path_hash = hashlib.md5(normalized_path.encode()).hexdigest()
    vector_path = os.path.join(VECTOR_DIR, path_hash)
    if os.path.exists(vector_path):
        return FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    try:
        loader = PyPDFLoader(pdf_path)
        texts = [p.page_content for p in loader.load()]
        vs = FAISS.from_texts(texts, embeddings)
        vs.save_local(vector_path)
        logger.info(f"✅ 벡터스토어 생성 완료: {pdf_path}")
        return vs
    except Exception as e:
        logger.error(f"❌ 벡터 생성 실패 ({pdf_path}): {e}")
        return None

def combine_vectorstores(pdf_paths: List[str]) -> Tuple[Optional[FAISS], Optional[str]]:
    vectorstores = []
    combined_content = ""
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            continue
        vs = get_or_create_vectorstore(pdf_path)
        if vs:
            vectorstores.append(vs)
            combined_content += extract_text_from_pdf(pdf_path) + "\n\n--- PDF 분리 ---\n\n"
    if not vectorstores:
        return None, None
    main_vs = vectorstores[0]
    for i in range(1, len(vectorstores)):
        main_vs.merge_from(vectorstores[i])
    return main_vs, combined_content.strip()

# ==========================================================
# 🧭 DB에서 PDF 경로 조회
# ==========================================================
def get_pdf_paths_for_content(content_id: int):
    connection = None
    paths = []
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_DATABASE,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
        with connection.cursor() as cursor:
            sql = "SELECT file_path FROM contents WHERE id=%s"
            cursor.execute(sql, (content_id,))
            result = cursor.fetchall()
            paths = [row["file_path"] for row in result if row["file_path"]]
    except Exception as e:
        logger.error(f"❌ DB 조회 실패: {e}")
    finally:
        if connection:
            connection.close()
    return paths

# ==========================================================
# 🩺 헬스체크
# ==========================================================
@app.get("/health/")
async def health_check():
    return {"status": "ok"}

# ==========================================================
# 📘 전체 요약 엔드포인트 (✅ /contents 로 prefix 통일)
# ==========================================================
@app.post("/contents/{content_id}/summarize")
async def summarize_full(content_id: int):
    pdf_paths = get_pdf_paths_for_content(content_id)
    if not pdf_paths:
        raise HTTPException(status_code=404, detail="PDF 경로 없음")

    _, combined_content = combine_vectorstores(pdf_paths)
    if not combined_content:
        raise HTTPException(status_code=400, detail="PDF 내용 로드 실패")

    prompt = f"""
다음 PDF 내용을 요약하세요.
핵심 개념 위주로 5문단 이하로 정리:
---
{combined_content[:8000]}
"""
    result = llm.invoke(prompt)
    return {"content_id": content_id, "summaryText": result.content.strip()}

# ==========================================================
# 📗 단원별 요약 (/contents/{id}/summaries)
# ==========================================================
@app.post("/contents/{content_id}/summaries")
async def summarize_chapter(content_id: int, request: Optional[ChapterRequest] = None):
    pdf_paths = request.pdf_paths if request and request.pdf_paths else get_pdf_paths_for_content(content_id)
    if not pdf_paths:
        raise HTTPException(status_code=404, detail="PDF 경로 없음")

    _, combined_content = combine_vectorstores(pdf_paths)
    if not combined_content:
        raise HTTPException(status_code=400, detail="PDF 내용 로드 실패")

    pattern = r"^(\d+\.\d+(\.\d+)?)\s*(.*)$"
    chapters = {}
    current_chapter = None
    for line in combined_content.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if match:
            current_chapter = match.group(1)
            chapters[current_chapter] = match.group(3) + "\n"
        elif current_chapter:
            chapters[current_chapter] += line + "\n"

    summaries = []
    if request and request.chapter_request:
        matched = [k for k in chapters.keys() if k.startswith(request.chapter_request)]
        combined = "\n".join([chapters[k] for k in matched])
        result = llm.invoke(f"다음 내용을 요약해줘:\n{combined[:8000]}")
        summaries.append({"chapter": request.chapter_request, "summaryText": result.content.strip()})
    else:
        for k, v in chapters.items():
            result = llm.invoke(f"{k} 내용을 요약해줘:\n{v[:8000]}")
            summaries.append({"chapter": k, "summaryText": result.content.strip()})

    return {"summaries": summaries}

# ==========================================================
# ❓ 질문 (RAG + 웹보조)
# ==========================================================
@app.post("/contents/{content_id}/ask")
async def ask_question(content_id: int, body: QuestionRequest):
    question = body.question
    force_web = body.force_web

    pdf_paths = get_pdf_paths_for_content(content_id)
    if not pdf_paths and not tavily_tool:
        raise HTTPException(status_code=404, detail="PDF 및 웹 검색 불가")

    vectorstore, combined_text = combine_vectorstores(pdf_paths) if pdf_paths else (None, None)
    do_force_web = force_web or any(k in question for k in ["웹", "검색", "인터넷"])

    if vectorstore and not do_force_web:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        response = rag_chain.invoke({"query": question})
        return {"answer": response.get("result", "결과 없음"), "source": "PDF"}

    if tavily_tool:
        resp = tavily_tool.invoke({"query": question})
        if isinstance(resp, dict) and "results" in resp:
            return {"answer": resp["results"][0].get("content", ""), "source": "WEB"}

    return {"answer": "관련 내용을 찾을 수 없습니다.", "source": "NONE"}

# ==========================================================
# 🧠 퀴즈 생성 (/contents/{id}/quiz/generate)
# ==========================================================
@app.post("/contents/{content_id}/quiz/generate")
async def quiz_generate(content_id: int, req: QuizGenerationRequest):
    pdf_paths = get_pdf_paths_for_content(content_id)
    if not pdf_paths:
        raise HTTPException(status_code=404, detail="PDF 경로 없음")

    _, content = combine_vectorstores(pdf_paths)
    if not content:
        raise HTTPException(status_code=400, detail="PDF 로드 실패")

    prompt = f"""
다음 내용을 바탕으로 {req.num_questions}개의 객관식 퀴즈를 생성하세요.
난이도: {req.difficulty}
출력 형식(JSON):
{{
"questions":[{{"question":"...", "options":["A","B","C","D"], "correct_answer":"...", "explanation":"..."}}]
}}
내용:
{content[:10000]}
"""
    result = llm.invoke(prompt)
    cleaned = result.content.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(cleaned)
    except:
        return {"raw_response": cleaned}
