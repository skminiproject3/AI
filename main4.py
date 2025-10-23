import os
import re
import hashlib
import json
import logging
from typing import List, Tuple, Optional, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain_tavily import TavilySearch
from fastapi import Form, Body
import pymysql
# -----------------------
# 환경 설정
# -----------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USER = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY 미설정!")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
tavily_tool = TavilySearch(api_key=TAVILY_API_KEY, max_results=3) if TAVILY_API_KEY else None

# -----------------------
# 로깅 설정
# -----------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# -----------------------
# 업로드 및 벡터 DB 폴더
# -----------------------
UPLOAD_DIR = "uploaded_pdfs"
VECTOR_DIR = "vector_stores"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# -----------------------
# Pydantic 모델
# -----------------------
class PdfPathsRequest(BaseModel):
    pdf_paths: List[str]

class ChapterRequest(BaseModel):
    pdf_paths: List[str]
    chapter_request: Optional[str] = None

class QuestionRequest(BaseModel):
    question: str
    pdf_paths: List[str]

class QuizGenerationRequest(BaseModel):
    pdf_paths: List[str]
    num_questions: int = Field(5, ge=1, le=20)
    difficulty: str = Field("MEDIUM", pattern="^(EASY|MEDIUM|HARD)$")
# -----------------------
# PDF 처리 및 벡터스토어
# -----------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return "\n".join([p.page_content for p in pages])
    except Exception as e:
        logger.error(f"PDF 추출 실패 ({pdf_path}): {e}")
        return ""

def get_or_create_vectorstore(pdf_path: str) -> Optional[FAISS]:
    normalized_path = pdf_path.replace("\\","/")
    path_hash = hashlib.md5(normalized_path.encode()).hexdigest()
    vector_path = os.path.join(VECTOR_DIR, path_hash)
    if os.path.exists(vector_path):
        return FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    try:
        loader = PyPDFLoader(pdf_path)
        texts = [p.page_content for p in loader.load()]
        if not texts:
            return None
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

def split_by_subchapter(text: str) -> Dict[str,str]:
    """텍스트에서 단원별로 분리 (숫자.숫자 또는 숫자.숫자.숫자)"""
    pattern = r"^(\d+\.\d+(\.\d+)?)\s*(.*)$"  # 숫자.숫자 + optional 단원 제목
    chapters = {}
    current_chapter = None
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if match:
            current_chapter = match.group(1)  # "4.3"만 키로 사용
            chapters[current_chapter] = match.group(3) + "\n"  # 제목 포함
        elif current_chapter:
            chapters[current_chapter] += line + "\n"
    return chapters
# PDF 경로를 boot에서 전달받는  함수 (Boot에서 DB 조회 후 전달)
# -----------------------
# PDF 경로 조회 함수
# -----------------------
def get_pdf_paths_for_content(content_id: int):
    """
    content_id 기준으로 MariaDB에서 file_path를 조회
    반환: 파일 경로 리스트
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
            sql = "SELECT file_path FROM contents WHERE id=%s"
            cursor.execute(sql, (content_id,))
            result = cursor.fetchall()
            paths = [row['file_path'] for row in result if row['file_path']]
    except Exception as e:
        print(f"❌ PDF 경로 조회 실패: {e}")
    finally:
        if connection:
            connection.close()

    return paths
# -----------------------
# 요약 프롬프트
# -----------------------
summary_prompt = PromptTemplate(
    input_variables=["content"],
    template="당신은 교재 핵심 요약 전문가입니다. 내용: {content}"
)

bulk_summary_prompt = PromptTemplate(
    input_variables=["content", "chapters_list"],
    template="교재 내용을 단원별 요약. chapters_list={chapters_list} 내용={content}"
)

# -----------------------
# FastAPI 초기화
# -----------------------
app = FastAPI(title="PDF 학습 도우미 API (Agent + RAG)")

# -----------------------
# 엔드포인트: Health
# -----------------------
@app.get("/health/")
async def health_check():
    return {"status": "ok"}

# -----------------------
# 엔드포인트: PDF 업로드
# -----------------------
@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    saved_files = []
    created_vectors = []
    for file in files:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        try:
            with open(save_path, "wb") as f:
                f.write(await file.read())
            saved_files.append(save_path)
            vs = get_or_create_vectorstore(save_path)
            if vs: created_vectors.append(save_path)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    return {
        "saved_files": saved_files,
        "vector_status": f"{len(created_vectors)}개의 파일 벡터 생성 완료.",
        "created_vectors_for": created_vectors
    }

# -----------------------
# 엔드포인트: 전체 요약
# -----------------------
@app.post("/api/contents/{content_id}/summarize")
async def summarize_full(content_id: int):
    """
    content_id: Boot DB contents.id
    Boot DB에서 PDF 경로를 가져와 벡터스토어에서 내용 로드 후 전체 요약
    """
    try:
        # 1️⃣ Boot DB에서 PDF 경로 가져오기
        pdf_paths = get_pdf_paths_for_content(content_id)
        if not pdf_paths:
            raise HTTPException(status_code=404, detail="PDF 경로가 없습니다")

        # 2️⃣ 벡터스토어 결합
        vectorstore, combined_content = combine_vectorstores(pdf_paths)
        if not combined_content:
            raise HTTPException(status_code=400, detail="PDF 내용 로드 실패")

        # 3️⃣ LLM으로 요약 생성
        prompt = f"""
다음 PDF 내용을 바탕으로 전체 요약을 작성하세요.
- 최대 5문단 내외
- 간결하고 핵심 위주
- 내용에 기반한 중요한 키워드 포함

내용:
{combined_content[:10000]}
"""
        result = llm.invoke(prompt)

        return {
            "content_id": content_id,
            "summaryText": result.content.strip()
        }

    except Exception as e:
        logger.error(f"전체 요약 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# 엔드포인트: 단원별 요약
# -----------------------
@app.post("/api/contents/{content_id}/summaries")
async def summarize_chapter(content_id: int, request: ChapterRequest):
    try:
        # 1️⃣ PDF 결합 (Boot DB에서 가져온 경로 사용)
        _, combined_content = combine_vectorstores(request.pdf_paths)
        if not combined_content:
            raise HTTPException(status_code=400, detail="PDF 내용 로드 실패")

        # 2️⃣ 단원별로 분리
        chapters = split_by_subchapter(combined_content)
        summaries = []

        if request.chapter_request:
            # 특정 단원 요청 시
            matched_keys = [k for k in chapters.keys() if k.startswith(request.chapter_request)]
            if matched_keys:
                combined = "\n\n".join([chapters[k] for k in matched_keys])
                prompt = f"""
다음 PDF 내용을 바탕으로 단원 요약을 작성하세요.
- 최대 3문단 내외
- 핵심 위주
- 내용에 기반한 중요한 키워드 포함

내용:
{combined[:10000]}
"""
                result = llm.invoke(prompt)
                summaries.append({
                    "chapter": ", ".join(matched_keys),
                    "summaryText": result.content.strip()
                })
        else:
            # 전체 단원 요약 요청 시
            chapters_list = sorted(chapters.keys())
            bulk_prompt_text = f"""
다음 PDF 내용을 바탕으로 각 단원별 요약을 작성하세요.
- 단원 순서대로
- 각 단원 최대 3문단
- 핵심 위주
내용:
{combined_content[:10000]}
단원 순서: {chapters_list}
"""
            result = llm.invoke(bulk_prompt_text)
            # LLM에서 각 단원별 요약을 줄 단위로 반환했다고 가정
            summaries = [{"chapter": k, "summaryText": v} for k,v in zip(chapters_list, result.content.strip().splitlines())]

        return {"summaries": summaries}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# LangChain Agent 기반 질문 엔드포인트
# -----------------------
@app.post("/api/contents/{content_id}/ask")
async def ask_question(content_id: int, question: str = Form(...), force_web: bool = Form(False)):
    """
    - content_id: Boot DB의 contents.id
    - question: form-data로 전송 (Boot에서 보낼 때 Key=question)
    - force_web: (옵션) True로 보내면 PDF 확인 없이 바로 웹 검색 수행
    """
    logger.info(f"질문 수신: content_id={content_id} / force_web={force_web} / question={question}")

    # 1) DB에서 PDF 경로 조회
    pdf_paths = get_pdf_paths_for_content(content_id)
    if not pdf_paths:
        # PDF가 없으면 웹만 시도 (가능하면)
        logger.warning("PDF 경로 없음: 웹 폴백 시도")
        if not tavily_tool:
            raise HTTPException(status_code=404, detail="PDF 및 웹 검색 불가 (파일 없음, Tavily 미설정)")
        pdf_paths = []

    # 2) vectorstore 준비 (있다면)
    vectorstore = None
    combined_text = None
    if pdf_paths:
        vectorstore, combined_text = combine_vectorstores(pdf_paths)
        if not vectorstore:
            logger.warning("Vectorstore 로드 실패 → 웹 폴백")

    # 3) force_web 체크: 사용자가 '웹에서' 같은 문자로 강제 요청 또는 폼 force_web True
    autotrig_web_keywords = any(k in question for k in ["웹에서", "인터넷", "검색", "웹으로"])
    do_force_web = force_web or autotrig_web_keywords

    # 4) PDF RAG 시도 (우선) — vectorstore 존재할 때만
    if vectorstore and not do_force_web:
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

            # RetrievalQA.invoke returns dict-like with 'result' and possibly 'source_documents' depending on version
            pdf_response = rag_chain.invoke({"query": question})
            logger.debug(f"PDF RAG 응답: keys={list(pdf_response.keys())}")

            # 안전하게 추출
            result_text = ""
            if isinstance(pdf_response, dict):
                result_text = str(pdf_response.get("result") or pdf_response.get("output_text") or "")
                sources = pdf_response.get("source_documents") or pdf_response.get("source_documents", []) or []
            else:
                # 일부 버전은 그냥 문자열 반환 가능
                result_text = str(pdf_response)
                sources = []

            # 판별: 근거 문서가 하나 이상 있고 결과 텍스트가 의미있어야 PDF 기반으로 판정
            if sources and len(sources) > 0 and len(result_text.strip()) > 10:
                # 추출한 소스의 간단 정보(길이 / 첫 문장) 리턴
                source_count = len(sources)
                # source_documents는 Document 객체일 가능성 있으므로 안전처리
                refs = []
                for s in sources[:3]:
                    try:
                        # Document 형식이면 .metadata / .page_content 존재
                        meta = getattr(s, "metadata", {}) or (s.get("metadata") if isinstance(s, dict) else {})
                        refs.append({
                            "metadata": meta,
                            "snippet": (getattr(s, "page_content", "") or s.get("page_content", "") if isinstance(s, dict) else "")[:300]
                        })
                    except Exception:
                        refs.append({"raw": str(s)[:300]})
                return {
                    "source": "PDF",
                    "content_id": content_id,
                    "question": question,
                    "answer": result_text.strip(),
                    "pdf_reference_count": source_count,
                    "pdf_references": refs,
                    "message": "PDF 문서에서 답변을 찾았습니다."
                }
            else:
                logger.info("PDF에서 유의미한 근거 못찾음 → 웹 폴백")
        except Exception as e:
            logger.exception(f"PDF RAG 처리 중 오류: {e}")
            # PDF 실패여도 웹으로 폴백

    # 5) 웹 폴백: Tavily 사용 (있을 때)
    if tavily_tool:
        try:
            # 사용자 입력에서 '웹에서', '검색' 등 제거해 정제된 쿼리 사용
            query_clean = re.sub(r"(웹에서|웹으로|인터넷|검색|해줘|알려줘)", "", question, flags=re.IGNORECASE).strip()
            if not query_clean:
                query_clean = question

            web_resp = tavily_tool.invoke({"query": query_clean})
            # web_resp 구조 예상: {'results': [ {url,title,content,score,...}, ... ], 'request_id':..., ...}
            results = web_resp.get("results", []) if isinstance(web_resp, dict) else []
            if not results:
                # 일부 버전은 키가 'results'가 아니고 바로 list 반환 가능
                if isinstance(web_resp, list):
                    results = web_resp

            if results:
                # score로 정렬 후 상위(최대 1개) 선택
                sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
                best = sorted_results[0]
                title = best.get("title") or best.get("url") or "출처 없음"
                url = best.get("url", "")
                extracted = best.get("content") or best.get("snippet") or ""

                # 만약 PDF(문서) 형식의 결과 (long content)가 있으면 LLM으로 요약
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
                    # 짧은 snippet일 경우 바로 반환
                    summary = best.get("content") or best.get("snippet") or str(best)[:400]

                return {
                    "source": "WEB",
                    "content_id": content_id,
                    "question": question,
                    "answer": summary,
                    "reference": {"title": title, "url": url},
                    "message": "웹 검색 결과를 기반으로 답변했습니다."
                }
            else:
                logger.info("웹 검색에서 결과 없음")
        except Exception as e:
            logger.exception(f"웹 검색 처리 중 오류: {e}")

    # 6) 실패: PDF도 웹도 못찾음
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

