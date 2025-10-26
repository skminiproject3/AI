import os
import re
import hashlib
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
#from aiohttp import request
from fastapi import FastAPI, UploadFile, File, HTTPException #, Form,Body
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
DB_USER = os.getenv("DB_USER")
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
    chapter: int
    #chapter_request: Optional[str] = Field(None, alias="chapterRequest")
class ChapterSummary(BaseModel):
    chapter: str
    summaryText: str

class SummariesResponse(BaseModel):
    summaries: List[ChapterSummary]
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
    
# ==========================================================
# ✅ 3️⃣ 벡터 DB에서 챕터별 내용 추출
# ==========================================================
def load_chapter_text_from_vector(vector_path: str, target_chapter: str) -> str:
    vs = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    
    combined_text = ""
    for doc in vs.docstore._dict.values():
        combined_text += getattr(doc, "page_content", "") + "\n"

    # metadata.json 순서 기반으로 다음 단원 위치 찾기
    metadata_path = os.path.join(vector_path, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    chapters = sorted(metadata.get("chapters", []), key=lambda x: [int(n) for n in x["chapter"].split(".")])
        # 타겟 챕터 메타데이터 찾기
    target_meta = next((ch for ch in chapters if ch.get("chapter") == target_chapter), None)
    
    if not target_meta:
        raise HTTPException(status_code=404, detail=f"metadata.json에 {target_chapter} 없음")
    target_title = target_meta.get("title", "").strip()
    WINDOW = 10  # 앞뒤 20글자 내에서 title 확인

    # --- 시작 위치 탐색 (이중 조건: 번호 + title proximity)
    start_idx = -1
    for m in re.finditer(re.escape(target_chapter), combined_text):
        idx = m.start()
        # 주변 20글자 내 확인
        context_start = max(0, idx - WINDOW)
        context_end = min(len(combined_text), idx + WINDOW)
        context = combined_text[context_start:context_end]
        if target_title and target_title in context:
            start_idx = idx
            break

    if start_idx == -1:
        raise HTTPException(status_code=404, detail=f"{target_chapter} 시작 위치를 찾을 수 없음 (번호와 제목이 근접하지 않음)")

    # 다음 단원 시작 위치
    next_idx = len(combined_text)
    for ch in chapters:
        if ch["chapter"] > target_chapter:
            idx = combined_text.find(ch["chapter"])
            if idx != -1:
                next_idx = idx
                break
    print(f"단원 텍스트 추출: {target_chapter} | start={start_idx} | end={next_idx}")
    
    return combined_text[start_idx:next_idx]


# ==========================================================
# ✅ 4️⃣ LLM을 이용해 요약 수행
# ==========================================================
def summarize_text_with_llm(chapter_label: str, text: str) -> str:
    """
    LangChain + OpenAI 모델을 사용해 텍스트 요약
    """
    prompt = f"다음 내용을 간결하게 요약해줘:\n\n{text[:12000]}"
    result = llm.invoke(prompt)
    return result.content.strip()

def get_or_create_vectorstore(pdf_path: str) -> Optional[FAISS]:
    normalized_path = pdf_path.replace("\\","/")
    path_hash = hashlib.md5(normalized_path.encode()).hexdigest()
    vector_path = os.path.join(VECTOR_DIR, path_hash)
    metadata_path = os.path.join(vector_path, "metadata.json")

    if os.path.exists(vector_path) and os.path.exists(metadata_path):
        logger.info(f"📂 기존 벡터스토어 및 metadata.json 로드: {pdf_path}")
        return FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)

    # PDF 텍스트 추출
    texts = extract_text_from_pdf(pdf_path)
    chapters = recognize_chapters_with_llm(texts)

    # 벡터스토어 생성
    vs = FAISS.from_texts([texts], embeddings)
    os.makedirs(vector_path, exist_ok=True)
    vs.save_local(vector_path)

    # metadata.json 저장
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"chapters": chapters}, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 벡터스토어 + metadata.json 생성 완료: {pdf_path}")

    return vs

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
# -----------------------
# 챕터 인식
# -----------------------
def recognize_chapters_with_llm(full_text: str):
    """
    첫 등장 숫자 기반 장 인식 후 LLM으로 검증
    """
    chapters = []
    first_match = re.search(r"^(\d+)\s", full_text, re.MULTILINE)
    if first_match:
        chapters.append({"chapter": first_match.group(1), "title": ""})

    # LLM으로 보정
    try:
        prompt = f"""
아래는 PDF 텍스트입니다.
텍스트에서 '장 번호'와 '장 제목'을 JSON 형태로 반환하세요.
예: {{ "chapters": [{{"chapter": "1", "title": "암호 개론"}}] }}

텍스트:
{full_text[:10000]}
"""
        resp = llm.invoke(prompt)
        json_text = resp.content.strip().replace("```json","").replace("```","")
        llm_chapters = json.loads(json_text).get("chapters", [])
        if llm_chapters:
            chapters = llm_chapters
    except Exception as e:
        logger.warning(f"LLM 챕터 인식 실패: {e}")

    return chapters

# ==========================================================
# 챕터 감지 로직
# ==========================================================
def detect_chapters_by_regex(text: str) -> List[str]:
    """
    PDF 텍스트에서 챕터 번호를 추출
    - 번호 형식: 1.1, 2.3.4, 제1장, CHAPTER 1 등
    - 중복 제거 후 정렬
    """
    try:
        chapters = set()
        # 1) 차례처럼 보이는 번호 우선 추출
        toc_matches = re.findall(r"\b\d+(?:\.\d+)+\b", text)
        if toc_matches:
            return sorted(toc_matches, key=lambda x: [int(n) for n in x.split('.')])
        
        # 2) 각 줄 스캔
        pattern = re.compile(r"(\b\d+(?:\.\d+)+\b|제\d+장|CHAPTER\s+\d+)", re.IGNORECASE)
        for line in text.splitlines():
            line = line.strip()
            match = pattern.search(line)
            if match:
                chapters.add(match.group(1))
        
        # 정렬: 숫자 기반 우선, 문자 포함 챕터는 뒤로
        def sort_key(ch):
            nums = re.findall(r"\d+", ch)
            return [int(n) for n in nums] if nums else [float('inf')]
        
        return sorted(chapters, key=sort_key)
    except Exception as e:
        logger.error(f"챕터 감지 실패 | {e}")
        return []

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
def get_vector_paths_for_content(content_id: int) -> List[str]:
    """
    content_id 기준으로 MariaDB에서 vector_path를 조회
    반환: 벡터스토어 경로 리스트
    """
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
            sql = "SELECT vector_path FROM contents WHERE id=%s"
            cursor.execute(sql, (content_id,))
            result = cursor.fetchall()
            paths = [row['vector_path'] for row in result if row['vector_path']]

    except Exception as e:
        print(f"❌ 벡터 경로 조회 실패: {e}")
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
def summarize_by_chapter(content_id: int, request: ChapterRequest):
    
    if not request.chapter:
        raise HTTPException(status_code=400, detail="chapter 필수")

    # 1️⃣ content_id 기준 벡터스토어 경로 조회
    vector_paths = get_vector_paths_for_content(content_id)
    if not vector_paths:
        raise HTTPException(status_code=404, detail="Vector path not found")

    vector_path = vector_paths[0]
    
    # 2️⃣ metadata.json 로드
    metadata_path = os.path.join(vector_path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="metadata.json 없음")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    chapters = metadata.get("chapters", [])
    if not chapters:
        raise HTTPException(status_code=404, detail="metadata.json에 단원이 없음")
    
    # 3️⃣ 마지막 숫자 기준 chapter 찾기
    target_chapter_obj = None
    for ch in chapters:
        parts = ch["chapter"].split(".")
        if parts[-1] == str(request.chapter):
            target_chapter_obj = ch
            
            break

    if not target_chapter_obj:
            return {
                "summaries": [
                    {
                        "chapter": str(request.chapter),
                        "title": "",
                        "summaryText": f" 요청 하신 chapter는 존재하지 않습니다."
                    }
                ]
            }
    
    target_chapter = target_chapter_obj["chapter"]
    chapter_title = target_chapter_obj.get("title", "")
    
    # 4️⃣ 벡터스토어에서 해당 chapter 텍스트 추출
    chapter_text = load_chapter_text_from_vector(vector_path, target_chapter)

    # 5️⃣ LLM으로 요약
    summary_text = summarize_text_with_llm(target_chapter, chapter_text)

    # 6️⃣ 결과 반환
    return {
        "summaries": [
            {
                "chapter": target_chapter,
                "title": chapter_title,
                "summaryText": summary_text
            }
        ]
    }


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