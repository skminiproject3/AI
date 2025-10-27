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
    allow_origins=origins,          # 개발 중이면 ["*"]도 가능(credential 안 쓸 때)
    allow_credentials=True,         # 쿠키/인증정보 쓰면 True
    allow_methods=["*"],            # 최소한 ["POST","GET","OPTIONS"]여도 됨
    allow_headers=["*"],            # 또는 ["content-type","authorization"]
    expose_headers=["*"],           # (선택) 클라에서 읽을 헤더
)

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
class QuestionRequest(BaseModel):
    question: str
    force_web: bool
    pdf_paths: Optional[List[str]] = None

class QuizGenerationRequest(BaseModel):
    pdf_paths: List[str]
    num_questions: int = Field(5, ge=1, le=20)
    difficulty: str = Field("MEDIUM", pattern="^(EASY|MEDIUM|HARD)$")
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
    combined_text_parts: List[str] = []

    for file in files:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        try:
            # 파일 저장
            with open(save_path, "wb") as f:
                data = await file.read()
                f.write(data)
            saved_files.append(save_path)
            logger.info("🗂️ 파일 저장 완료 | path=%s | size=%d bytes", save_path, len(data))

            # 텍스트 추출
            text = extract_text_from_pdf(save_path)
            if text.strip():
                combined_text_parts.append(text)
            else:
                logger.warning("⚠️ 텍스트 추출 실패 또는 빈 문서 | path=%s", save_path)

        except Exception as e:
            logger.error("❌ 업로드 실패 | file=%s | error=%s", file.filename, e)
            raise HTTPException(status_code=500, detail=f"파일 업로드 실패: {e}")

    if not combined_text_parts:
        raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출하지 못했습니다.")

    # 전체 문서 기준 챕터 감지
    combined_text = "\n\n--- PDF 분리 ---\n\n".join(combined_text_parts)
    detection = get_total_chapters(combined_text)

    total_chapters_result: int = int(detection.get("total_chapters", 1))
    detected_method: str = str(detection.get("method", "none"))
    all_chapters: List[str] = list(detection.get("chapter_list", []))

    # 벡터스토어 생성 및 vector_path 계산 (무조건 생성)
    vector_paths: List[str] = []
    for file_path in saved_files:
        vs = get_or_create_vectorstore(file_path)
        normalized_path = file_path.replace("\\", "/")
        path_hash = hashlib.md5(normalized_path.encode()).hexdigest()
        vector_path = os.path.join(VECTOR_DIR, path_hash)
        vector_paths.append(vector_path)

    # 기본적으로 하나만 반환 (boot 연동용)
    final_vector_path = vector_paths[0] if vector_paths else None
    logger.info(
        "✅ 업로드/분석 완료 | files=%d | total_chapters=%d | method=%s | vector_path=%s",
        len(saved_files), total_chapters_result, detected_method, final_vector_path
    )

    return {
        "pdf_paths": saved_files,
        "total_chapters": total_chapters_result,
        "method": detected_method,
        "chapter_list": all_chapters,
        "vector_path": final_vector_path
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
    # pdf_paths 유효성 확인
    pdf_paths = []
    if request and request.pdf_paths:
        pdf_paths = request.pdf_paths
    else:
        pdf_paths = get_pdf_paths_for_content(content_id)  # ✅ DB에서 가져오기

    if not pdf_paths:
        raise HTTPException(status_code=404, detail="PDF 경로를 찾을 수 없습니다.")

    # chapter_request 읽기 (없으면 전체 요약)
    chapter_req = None
    if request and request.chapter_request:
        chapter_req = request.chapter_request

    # PDF 내용 병합
    _, combined_content = combine_vectorstores(pdf_paths)
    if not combined_content:
        raise HTTPException(status_code=400, detail="PDF 내용 로드 실패")

    # 챕터별 요약 생성 (이전 로직 동일)
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

# -----------------------
# LangChain Agent 기반 질문 엔드포인트
# -----------------------
@app.post("/api/contents/{content_id}/ask")
async def ask_question(content_id: int, body: QuestionRequest):
    """
    - content_id: Boot DB의 contents.id
    - question: JSON body로 전송 {"question": "...", "force_web": false}
    - force_web: (옵션) True로 보내면 PDF 확인 없이 바로 웹 검색 수행
    """
    question = body.question
    force_web = body.force_web

    logger.info(f"질문 수신: content_id={content_id} / force_web={force_web} / question={question}")

    # DB에서 PDF 경로 조회
    pdf_paths = get_pdf_paths_for_content(content_id)
    if not pdf_paths:
        logger.warning("PDF 경로 없음: 웹 폴백 시도")
        if not tavily_tool:
            raise HTTPException(status_code=404, detail="PDF 및 웹 검색 불가 (파일 없음, Tavily 미설정)")
        pdf_paths = []

    # ectorstore 준비
    vectorstore = None
    combined_text = None
    if pdf_paths:
        vectorstore, combined_text = combine_vectorstores(pdf_paths)
        if not vectorstore:
            logger.warning("Vectorstore 로드 실패 → 웹 폴백")

    # force_web 또는 “웹에서/검색” 키워드 자동 감지
    autotrig_web_keywords = any(k in question for k in ["웹에서", "인터넷", "검색", "웹으로"])
    do_force_web = force_web or autotrig_web_keywords

    # PDF RAG 시도 (force_web=False 일 때만)
    if vectorstore and not do_force_web:
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )

            pdf_response = rag_chain.invoke({"query": question})
            logger.debug(f"PDF RAG 응답: keys={list(pdf_response.keys()) if isinstance(pdf_response, dict) else 'N/A'}")

            # PDF 결과 정리
            result_text = ""
            sources = []
            if isinstance(pdf_response, dict):
                result_text = str(pdf_response.get("result") or pdf_response.get("output_text") or "")
                sources = pdf_response.get("source_documents") or []
            else:
                result_text = str(pdf_response)

            # 결과 검증
            if sources and len(result_text.strip()) > 10:
                refs = []
                for s in sources[:3]:
                    try:
                        meta = getattr(s, "metadata", {}) or (s.get("metadata") if isinstance(s, dict) else {})
                        snippet = (getattr(s, "page_content", "") or s.get("page_content", ""))[:300]
                        refs.append({"metadata": meta, "snippet": snippet})
                    except Exception:
                        refs.append({"raw": str(s)[:300]})
                return {
                    "source": "PDF",
                    "content_id": content_id,
                    "question": question,
                    "answer": result_text.strip(),
                    "pdf_reference_count": len(sources),
                    "pdf_references": refs,
                    "message": "PDF 문서에서 답변을 찾았습니다."
                }
            else:
                logger.info("PDF에서 유의미한 근거 못찾음 → 웹 폴백")
        except Exception as e:
            logger.exception(f"PDF RAG 처리 중 오류: {e}")
            # 실패시 웹검색으로 넘어감

    # 웹 폴백 (Tavily 검색)
    if tavily_tool:
        try:
            query_clean = re.sub(r"(웹에서|웹으로|인터넷|검색|해줘|알려줘)", "", question, flags=re.IGNORECASE).strip()
            if not query_clean:
                query_clean = question

            web_resp = tavily_tool.invoke({"query": query_clean})
            results = web_resp.get("results", []) if isinstance(web_resp, dict) else web_resp

            if results:
                sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
                best = sorted_results[0]
                title = best.get("title") or best.get("url") or "출처 없음"
                url = best.get("url", "")
                extracted = best.get("content") or best.get("snippet") or ""

                if extracted and len(extracted) > 30:
                    prompt = f"""
아래 출처 정보를 바탕으로 질문에 대한 간결하고 정확한 한국어 답변을 작성하세요.
출처 제목: {title}
출처 URL: {url}
출처 내용:
{extracted}

질문: {question}

- 답변은 3~6 문장 이내로 명확하게 작성하세요.
- 출처 URL을 'reference' 필드에 포함시키세요.
"""
                    summary = llm.invoke(prompt).content.strip()
                else:
                    summary = extracted or str(best)[:400]

                return {
                    "source": "WEB",
                    "content_id": content_id,
                    "question": question,
                    "answer": summary,
                    "reference": {"title": title, "url": url},
                    "message": "웹 검색 결과를 기반으로 답변했습니다."
                }

            logger.info("웹 검색 결과 없음")

        except Exception as e:
            logger.exception(f"웹 검색 처리 중 오류: {e}")

    # 모든 경로 실패 시
    return {
        "source": "NONE",
        "content_id": content_id,
        "question": question,
        "answer": "관련 정보를 PDF 및 웹에서 찾을 수 없습니다.",
        "message": "검색 실패 또는 외부 의존성 미설정"
    }

# -----------------------
# 퀴즈 생성
# -----------------------
@app.post("/api/contents/{content_id}/quiz/generate")
async def quiz_generate(content_id: int, request: dict):
    try:
        # Boot에서 전달한 데이터 가져오기
        num_questions = request.get("num_questions")
        difficulty = request.get("difficulty")

        if not num_questions or not difficulty:
            raise HTTPException(status_code=400, detail="num_questions, difficulty 필수")

        # ✅ Boot-DB에서 PDF 경로 가져오기
        saved_pdfs = get_pdf_paths_for_content(content_id)
        if not saved_pdfs:
            raise HTTPException(status_code=404, detail="PDF 경로 없음")

        # ✅ PDF에서 벡터스토어 로딩
        vectorstore, content = combine_vectorstores(saved_pdfs)
        if not content:
            raise HTTPException(status_code=400, detail="PDF 로드 실패")

        # ✅ LLM 요청 Prompt
        prompt = f"""
당신은 시험 문제를 만드는 전문가입니다.
다음 내용을 바탕으로 객관식 퀴즈 {num_questions}개 생성하세요.
난이도: {difficulty}

출력 형식(JSON):
{{
"questions":[
    {{
    "question": "문제",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "정답",
    "explanation": "정답 해설"
    }}
]}}
내용:
{content[:12000]}
"""

        result = llm.invoke(prompt)

        try:
            cleaned = result.content.strip().replace("```json","").replace("```","")
            data = json.loads(cleaned)
            return data  # ✅ Boot가 원하는 구조 그대로 반환

        except Exception as e:
            logger.error(f"JSON 파싱 실패 → 원본 반환: {e}")
            return {"raw_response": result.content.strip()}

    except Exception as e:
        logger.error(f"퀴즈 생성 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))