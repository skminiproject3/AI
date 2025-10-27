import os
import re
import hashlib
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_tavily import TavilySearch
import pymysql

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ==========================================================
# 환경 설정 및 초기화
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

app = FastAPI(title="PDF 학습 도우미 API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ==========================================================
# 🔧 업로드 경로 자동 분기 (로컬/배포 모두 대응)
# ==========================================================
if os.getenv("ENV_MODE", "local") == "prod":
    UPLOAD_DIR = "/uploads"  # ✅ Docker(배포) 환경
else:
    UPLOAD_DIR = os.path.join(os.getcwd(), "uploaded_pdfs")  # ✅ 로컬 실행 시

VECTOR_DIR = "vector_stores"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ==========================================================
# Pydantic 모델 정의
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
# PDF 유틸 함수
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
# 챕터 감지
# ==========================================================
def detect_chapters_by_regex(text: str) -> List[str]:
    patterns = [
        r"제\s*\d+\s*장",
        r"CHAPTER\s+\d+",
        r"Chapter\s+\d+",
        r"\b\d+\.\d+",
        r"Section\s+\d+",
        r"Part\s+[IVXLC\d]+"
    ]
    found = set()
    for p in patterns:
        matches = re.findall(p, text)
        for m in matches:
            found.add(m.strip())
    return sorted(found)

def detect_chapters_by_llm(text: str) -> List[str]:
    prompt = f"""
다음 텍스트의 챕터 제목을 순서대로 뽑아주세요.
예시 출력: ["제1장 개요", "제2장 이론"]
---
{text[:4000]}
"""
    try:
        response = llm.invoke(prompt)
        lines = re.findall(r"제?\s*\d+\s*장|CHAPTER\s+\d+|Chapter\s+\d+|\d+\.\d+", response.content)
        return sorted(set(lines))
    except Exception as e:
        logger.warning(f"⚠️ LLM 챕터 감지 실패: {e}")
        return []

def get_total_chapters(text: str) -> Dict[str, Any]:
    regex_chapters = detect_chapters_by_regex(text)
    if regex_chapters:
        return {"method": "regex", "total_chapters": len(regex_chapters), "chapter_list": regex_chapters}
    llm_chapters = detect_chapters_by_llm(text)
    if llm_chapters:
        return {"method": "llm", "total_chapters": len(llm_chapters), "chapter_list": llm_chapters}
    return {"method": "none", "total_chapters": 1, "chapter_list": []}

# ==========================================================
# DB 조회
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
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        with connection.cursor() as cursor:
            sql = "SELECT file_path FROM contents WHERE id=%s"
            cursor.execute(sql, (content_id,))
            result = cursor.fetchall()
            paths = [row['file_path'] for row in result if row['file_path']]
    except Exception as e:
        logger.error(f"❌ DB 조회 실패: {e}")
    finally:
        if connection:
            connection.close()
    return paths

# ==========================================================
# FastAPI 엔드포인트
# ==========================================================
app = FastAPI(title="PDF 학습 도우미 API (Full 통합)")

@app.get("/health/")
async def health_check():
    return {"status": "ok"}

# 이하 나머지 엔드포인트는 그대로 유지
# /upload_pdfs, /summarize, /ask, /quiz/generate 등...
