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

# ==========================================================
# 환경 설정 및 초기화
# ==========================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USER = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY 미설정!")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
tavily_tool = TavilySearch(api_key=TAVILY_API_KEY, max_results=3) if TAVILY_API_KEY else None

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploaded_pdfs"
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

class QuizGenerationRequest(BaseModel):
    pdf_paths: List[str]
    num_questions: int = Field(5, ge=1, le=20)
    difficulty: str = Field("MEDIUM", pattern="^(EASY|MEDIUM|HARD)$")

# ==========================================================
# PDF → 텍스트 및 벡터화
# ==========================================================
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return "\n".join([p.page_content for p in pages])
    except Exception as e:
        logger.error(f"PDF 추출 실패 ({pdf_path}): {e}")
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
# 챕터 감지 로직
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
# DB에서 content_id로 PDF 경로 조회
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
# FastAPI 초기화
# ==========================================================
app = FastAPI(title="PDF 학습 도우미 API (Full 통합)")

# ==========================================================
# ✅ /health
# ==========================================================
@app.get("/health/")
async def health_check():
    return {"status": "ok"}

# ==========================================================
# ✅ /upload_pdfs
# ==========================================================
@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="업로드된 파일이 없습니다.")

    logger.info("📥 업로드 요청 수신 | files=%d", len(files))

    saved_files: List[str] = []
    created_vectors_for: List[str] = []
    combined_text_parts: List[str] = []

    for file in files:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        try:
            # 1) 파일 저장
            with open(save_path, "wb") as f:
                data = await file.read()
                f.write(data)
            saved_files.append(save_path)
            logger.info("🗂️ 파일 저장 완료 | path=%s | size=%d bytes", save_path, len(data))

            # 2) 텍스트 추출 (챕터 감지용)
            text = extract_text_from_pdf(save_path)
            if not text.strip():
                logger.warning("⚠️ 텍스트 추출 실패 또는 빈 문서 | path=%s", save_path)
            else:
                combined_text_parts.append(text)

            # 3) 벡터 생성/로드
            vs = get_or_create_vectorstore(save_path)
            if vs:
                created_vectors_for.append(save_path)
                logger.info("🧠 벡터 생성/로드 완료 | path=%s", save_path)
            else:
                logger.warning("⚠️ 벡터 생성 실패 | path=%s", save_path)

        except Exception as e:
            logger.error("❌ 업로드 실패 | file=%s | error=%s", file.filename, e)
            raise HTTPException(status_code=500, detail=f"파일 업로드 실패: {e}")

    # 4) 전체 문서 기준으로 챕터 감지 (여러 파일이면 합쳐서 감지)
    if not combined_text_parts:
        raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출하지 못했습니다.")

    combined_text = "\n\n--- PDF 분리 ---\n\n".join(combined_text_parts)
    detection = get_total_chapters(combined_text)

    total_chapters_result: int = int(detection.get("total_chapters", 1))
    detected_method: str = str(detection.get("method", "none"))
    all_chapters: List[str] = list(detection.get("chapter_list", []))

    logger.info(
        "✅ 업로드/분석 완료 | files=%d | vectors=%d | total_chapters=%d | method=%s",
        len(saved_files), len(created_vectors_for), total_chapters_result, detected_method
    )

    return {
        "pdf_paths": saved_files,
        "total_chapters": total_chapters_result,
        "method": detected_method,
        "chapter_list": all_chapters
    }
# ==========================================================
# ✅ /summarize (전체 요약)
# ==========================================================
@app.post("/api/contents/{content_id}/summarize")
async def summarize_full(content_id: int):
    pdf_paths = get_pdf_paths_for_content(content_id)
    if not pdf_paths:
        raise HTTPException(status_code=404, detail="PDF 경로 없음")
    _, combined_content = combine_vectorstores(pdf_paths)
    if not combined_content:
        raise HTTPException(status_code=400, detail="PDF 내용 로드 실패")
    prompt = f"""
다음 PDF 내용을 바탕으로 전체 요약을 작성하세요.
핵심 위주, 5문단 이하.
---
{combined_content[:10000]}
"""
    result = llm.invoke(prompt)
    return {"content_id": content_id, "summaryText": result.content.strip()}

# ==========================================================
# ✅ /summaries (단원별 요약)
# ==========================================================
@app.post("/api/contents/{content_id}/summaries")
async def summarize_chapter(content_id: int, request: Optional[ChapterRequest] = None):
    """
    pdf_paths를 안 보낸 경우, DB에서 자동으로 해당 content_id의 PDF 경로 조회
    """
    # 1️⃣ pdf_paths 유효성 확인
    pdf_paths = []
    if request and request.pdf_paths:
        pdf_paths = request.pdf_paths
    else:
        pdf_paths = get_pdf_paths_for_content(content_id)  # ✅ DB에서 가져오기

    if not pdf_paths:
        raise HTTPException(status_code=404, detail="PDF 경로를 찾을 수 없습니다.")

    # 2️⃣ chapter_request 읽기 (없으면 전체 요약)
    chapter_req = None
    if request and request.chapter_request:
        chapter_req = request.chapter_request

    # 3️⃣ PDF 내용 병합
    _, combined_content = combine_vectorstores(pdf_paths)
    if not combined_content:
        raise HTTPException(status_code=400, detail="PDF 내용 로드 실패")

    # 4️⃣ 챕터별 요약 생성 (이전 로직 동일)
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
    if chapter_req:
        matched = [k for k in chapters.keys() if k.startswith(chapter_req)]
        combined = "\n".join([chapters[k] for k in matched])
        result = llm.invoke(f"다음 내용을 요약해줘:\n{combined[:8000]}")
        summaries.append({"chapter": chapter_req, "summaryText": result.content.strip()})
    else:
        for k, v in chapters.items():
            result = llm.invoke(f"{k} 내용을 요약해줘:\n{v[:8000]}")
            summaries.append({"chapter": k, "summaryText": result.content.strip()})

    return {"summaries": summaries}

# ==========================================================
# ✅ /ask (질의응답)
# ==========================================================
@app.post("/api/contents/{content_id}/ask")
async def ask_question(content_id: int, question: str = Form(...)):
    pdf_paths = get_pdf_paths_for_content(content_id)
    vectorstore, _ = combine_vectorstores(pdf_paths)
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        resp = rag_chain.invoke({"query": question})
        if isinstance(resp, dict):
            answer = resp.get("result") or resp.get("output_text")
        else:
            answer = str(resp)
        if answer:
            return {"source": "PDF", "answer": answer.strip()}
    if tavily_tool:
        web_resp = tavily_tool.invoke({"query": question})
        results = web_resp.get("results", [])
        if results:
            top = results[0]
            summary = llm.invoke(f"아래 내용을 바탕으로 답변 작성:\n{top.get('content','')[:2000]}\nQ:{question}")
            return {"source": "WEB", "answer": summary.content.strip()}
    return {"source": "NONE", "answer": "관련 정보를 찾을 수 없습니다."}

# ==========================================================
# ✅ /quiz/generate (퀴즈 생성)
# ==========================================================
@app.post("/api/contents/{content_id}/quiz/generate")
async def quiz_generate(content_id: int, request: QuizGenerationRequest):
    pdf_paths = get_pdf_paths_for_content(content_id)
    _, content = combine_vectorstores(pdf_paths)
    if not content:
        raise HTTPException(status_code=400, detail="PDF 내용 없음")
    prompt = f"""
당신은 교재 기반 객관식 문제 생성 전문가입니다.
다음 내용을 기반으로 {request.num_questions}개의 문제를 생성하세요.
난이도: {request.difficulty}
형식(JSON):
{{
"questions":[{{"question":"...","options":["A","B","C","D"],"correct_answer":"정답","explanation":"이유"}}]
}}
내용:
{content[:12000]}
"""
    result = llm.invoke(prompt)
    try:
        cleaned = result.content.replace("```json","").replace("```","").strip()
        data = json.loads(cleaned)
        return data
    except:
        return {"raw_response": result.content}
