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
class AskRequest(BaseModel):
    question: str
class ChapterRequest(BaseModel):
    pdf_paths: Optional[List[str]] = None
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

    return 
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

def query_web(question: str):
    """
    웹 검색 후 LLM으로 학습 관련 정보 요약
    """
    try:
        # 질문에서 "웹", "검색" 등 키워드 제거
        query_clean = re.sub(r"(웹에서|웹으로|인터넷|검색|알려줘|해줘)", "", question, flags=re.IGNORECASE).strip()
        if not query_clean:
            query_clean = question

        web_resp = tavily_tool.invoke({"query": query_clean})
        results = web_resp.get("results", []) if isinstance(web_resp, dict) else (web_resp if isinstance(web_resp, list) else [])

        if not results:
            return None

        # 최고 점수 결과 선택
        best = sorted(results, key=lambda x: x.get("score", 0), reverse=True)[0]
        title = best.get("title") or best.get("url") or "출처 없음"
        url = best.get("url") or ""
        snippet = best.get("content") or best.get("snippet") or str(best)[:400]

        # LLM으로 학습 관련 정보만 요약
        summary_prompt = f"""
아래 출처 정보를 기반으로 질문에 대한 간결하고 정확한 답변을 작성하세요.
- 답변은 학습 관련 정보 중심으로 작성
출처 제목: {title}
출처 URL: {url}
내용: {snippet}
질문: {question}
"""
        answer = llm.invoke(summary_prompt).content.strip()
        if url:
            answer += f"\n\n출처: {url}"

        return answer
    except Exception as e:
        logger.exception(f"웹 검색 처리 중 오류: {e}")
        return None

def query_pdf_rag(vectorstore, question: str) -> str:
    """
    PDF RAG 기반 답변
    """
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        pdf_response = rag_chain.invoke({"query": question})

        # PDF 결과 안전하게 추출
        if isinstance(pdf_response, dict):
            result_text = str(pdf_response.get("result") or pdf_response.get("output_text") or "")
            sources = pdf_response.get("source_documents") or []
        else:
            result_text = str(pdf_response)
            sources = []

        if sources and len(result_text.strip()) > 10:
            return result_text.strip()
        else:
            return None
    except Exception as e:
        logger.exception(f"PDF RAG 처리 중 오류: {e}")
        return None
    
async def ask_question_logic(content_id: int, question: str, force_web: bool) -> Dict:
    """
    content_id: Boot DB contents.id
    question: 질문 텍스트
    force_web: True이면 PDF 무시하고 웹 검색 강제
    반환: dict {"source":"PDF"/"WEB"/"NONE", "answer":..., "reference":..., "message":...}
    """

    # 1) PDF 경로 조회
    pdf_paths = get_pdf_paths_for_content(content_id)
    vectorstore, combined_text = (None, None)
    if pdf_paths:
        vectorstore, combined_text = combine_vectorstores(pdf_paths)

    # 2) PDF 기반 RAG
    if vectorstore and not force_web:
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
            pdf_resp = rag_chain.invoke({"query": question})

            # PDF 응답 파싱
            answer_text = ""
            sources = []
            if isinstance(pdf_resp, dict):
                answer_text = str(pdf_resp.get("result") or pdf_resp.get("output_text") or "")
                sources = pdf_resp.get("source_documents") or []
            else:
                answer_text = str(pdf_resp)

            if sources and len(answer_text.strip()) > 10:
                refs = []
                for s in sources[:3]:
                    meta = getattr(s, "metadata", {}) or (s.get("metadata") if isinstance(s, dict) else {})
                    snippet = getattr(s, "page_content", "") or s.get("page_content", "")
                    refs.append({"metadata": meta, "snippet": snippet[:300]})
                return {"source": "PDF", "answer": answer_text.strip(), "pdf_references": refs, "message": "PDF 문서에서 답변을 찾았습니다."}
        except Exception as e:
            logger.exception(f"PDF RAG 처리 실패: {e}")

    # 3) 웹 폴백 (Tavily)
    if tavily_tool:
        try:
            query_clean = re.sub(r"(웹에서|웹으로|인터넷|검색|해줘|알려줘)", "", question, flags=re.IGNORECASE).strip()
            query_clean = query_clean or question
            web_resp = tavily_tool.invoke({"query": query_clean})
            results = web_resp.get("results", []) if isinstance(web_resp, dict) else web_resp

            if results:
                best = sorted(results, key=lambda x: x.get("score",0), reverse=True)[0]
                title = best.get("title") or best.get("url") or "출처 없음"
                url = best.get("url", "")
                content = best.get("content") or best.get("snippet") or str(best)[:400]

                if len(content) > 30:
                    prompt = f"""
아래 정보를 바탕으로 질문에 대한 간결하고 정확한 답변 작성 (한국어)
출처 제목: {title}
출처 URL: {url}
출처 내용: {content}
질문: {question}
- 답변 3~6문장
- reference에 URL 포함
"""
                    summary = llm.invoke(prompt).content.strip()
                else:
                    summary = content

                return {"source": "WEB", "answer": summary, "reference": {"title": title, "url": url}, "message": "웹 검색 결과 기반 답변"}
        except Exception as e:
            logger.exception(f"웹 검색 실패: {e}")

    # 4) 실패
    return {"source": "NONE", "answer": "PDF 및 웹에서 관련 정보 없음", "message": "검색 실패"}
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
@app.post("/api/contents/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    response_list = []

    for file in files:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        try:
            # 1️⃣ 파일 저장
            with open(save_path, "wb") as f:
                f.write(await file.read())

            # 2️⃣ 벡터스토어 생성
            vs = get_or_create_vectorstore(save_path)
            vector_path = None
            if vs:
                # vector_path 계산
                normalized_path = save_path.replace("\\", "/")
                path_hash = hashlib.md5(normalized_path.encode()).hexdigest()
                vector_path = os.path.join(VECTOR_DIR, path_hash)

            # 3️⃣ Boot에서 기대하는 형태로 객체 생성
            response_list.append({
                "contentId": None,  # Boot에서 DB 저장 후 채워야 하는 경우
                "title": file.filename,
                "status": "COMPLETED",
                "vectorId": None,
                "vector_path": vector_path
            })

        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return response_list

# -----------------------
# 엔드포인트: 전체 요약
# -----------------------
@app.post("/api/contents/{content_id}/summarize")
async def summarize_full(content_id: int):
    """
    content_id: Boot DB contents.id
    Boot DB에서 vector_path를 가져와 벡터스토어에서 내용 로드 후 전체 요약
    """
    try:
        # 1️⃣ DB에서 vector_path 조회
        vector_paths = get_vector_paths_for_content(content_id)
        if not vector_paths:
            raise HTTPException(status_code=404, detail="벡터스토어 경로가 없습니다")

        # 2️⃣ vector_path 기반 FAISS 로드
        vectorstores = []
        for vp in vector_paths:
            if os.path.exists(vp):
                try:
                    vs = FAISS.load_local(vp, embeddings, allow_dangerous_deserialization=True)
                    vectorstores.append(vs)
                except Exception as e:
                    logger.error(f"벡터스토어 로드 실패 ({vp}): {e}")

        if not vectorstores:
            raise HTTPException(status_code=500, detail="벡터스토어 로드 실패")

        # 3️⃣ 여러 벡터스토어가 있으면 합치기
        main_vs = vectorstores[0]
        for vs in vectorstores[1:]:
            main_vs.merge_from(vs)

        # 4️⃣ 벡터스토어에서 전체 텍스트 확보
        retriever = main_vs.as_retriever(search_kwargs={"k": 5})
        rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        pdf_resp = rag_chain.invoke({"query": "전체 내용 요약"})  # 전체 내용 확보용
        if isinstance(pdf_resp, dict):
            combined_content = str(pdf_resp.get("result") or pdf_resp.get("output_text") or "")
        else:
            combined_content = str(pdf_resp)

        if not combined_content or len(combined_content.strip()) < 20:
            raise HTTPException(status_code=500, detail="벡터스토어 내용 부족")

        # 5️⃣ LLM으로 요약 생성
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
        # 1️⃣ DB에서 vector_path 조회
        vector_paths = get_vector_paths_for_content(content_id)
        if not vector_paths:
            raise HTTPException(status_code=404, detail="벡터스토어 경로가 없습니다")

        # 2️⃣ vector_path 기반 FAISS 로드 및 전체 텍스트 합치기
        vectorstores = []
        combined_content = ""
        for vp in vector_paths:
            if os.path.exists(vp):
                try:
                    vs = FAISS.load_local(vp, embeddings, allow_dangerous_deserialization=True)
                    vectorstores.append(vs)

                    # FAISS에서 직접 텍스트 추출
                    for doc in vs.docstore._dict.values():
                        page_text = getattr(doc, "page_content", "") if hasattr(doc, "page_content") else str(doc)
                        combined_content += page_text + "\n\n--- PDF 분리 ---\n\n"

                except Exception as e:
                    logger.error(f"벡터스토어 로드 실패 ({vp}): {e}")

        if not vectorstores or not combined_content.strip():
            raise HTTPException(status_code=500, detail="벡터스토어 로드 또는 텍스트 확보 실패")

        # 3️⃣ 여러 벡터스토어가 있으면 합치기
        main_vs = vectorstores[0]
        for vs in vectorstores[1:]:
            main_vs.merge_from(vs)

        # 4️⃣ 단원별로 분리
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
{combined[:12000]}
"""
                result = llm.invoke(prompt)
                summaries.append({
                    "chapter": ", ".join(matched_keys),
                    "summaryText": result.content.strip()
                })
        else:
            # 전체 단원 요약 요청 시
            chapters_list = sorted(chapters.keys())
            for k in chapters_list:
                text = chapters[k][:12000]
                prompt = f"""
다음 PDF 내용을 바탕으로 단원 요약을 작성하세요.
- 최대 3문단 내외
- 핵심 위주
- 내용에 기반한 중요한 키워드 포함

내용:
{text}
"""
                result = llm.invoke(prompt)
                summaries.append({
                    "chapter": k,
                    "summaryText": result.content.strip()
                })

        return {"summaries": summaries}

    except Exception as e:
        logger.error(f"단원별 요약 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# LangChain Agent 기반 질문 엔드포인트
# -----------------------
@app.post("/api/contents/{content_id}/ask")
async def ask_question(content_id: int, req: AskRequest):
    question = req.question
    # 1️⃣ DB에서 vector_path 조회
    vector_paths = get_vector_paths_for_content(content_id)
    vectorstore = None
    answer = None
    if vector_paths:
        # 여러 벡터스토어가 있으면 합치기
        vectorstores = []
        for vp in vector_paths:
            if os.path.exists(vp):
                try:
                    vs = FAISS.load_local(vp, embeddings, allow_dangerous_deserialization=True)
                    vectorstores.append(vs)
                except Exception as e:
                    logger.error(f"벡터스토어 로드 실패 ({vp}): {e}")
        if vectorstores:
            main_vs = vectorstores[0]
            for vs in vectorstores[1:]:
                main_vs.merge_from(vs)
            vectorstore = main_vs


    # 웹 폴백 여부 판단
    do_force_web = any(k in question for k in ["웹에서", "검색", "인터넷", "알려줘", "웹으로"])
    if not answer or do_force_web:
        web_answer = query_web(question)
        if web_answer:
            answer = web_answer

    # 최종 실패 처리
    if not answer:
        answer = "관련 정보를 PDF 및 웹에서 찾을 수 없습니다."

    return answer

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

        # 1️⃣ DB에서 vector_path 조회
        vector_paths = get_vector_paths_for_content(content_id)
        if not vector_paths:
            raise HTTPException(status_code=404, detail="벡터스토어 경로 없음")

        # 2️⃣ vector_path 기반 FAISS 로드
        vectorstores = []
        for vp in vector_paths:
            if os.path.exists(vp):
                try:
                    vs = FAISS.load_local(vp, embeddings, allow_dangerous_deserialization=True)
                    vectorstores.append(vs)
                except Exception as e:
                    logger.error(f"벡터스토어 로드 실패 ({vp}): {e}")

        if not vectorstores:
            raise HTTPException(status_code=500, detail="벡터스토어 로드 실패")
        # 3️⃣ 여러 벡터스토어가 있으면 합치기
        main_vs = vectorstores[0]
        for vs in vectorstores[1:]:
            main_vs.merge_from(vs)
            
        # 4️⃣ 벡터스토어에서 텍스트 추출 (검색용)
        retriever = main_vs.as_retriever(search_kwargs={"k": 5})
        rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        pdf_resp = rag_chain.invoke({"query": "전체 내용 요약"})  # 전체 내용 확보용
        if isinstance(pdf_resp, dict):
            content = str(pdf_resp.get("result") or pdf_resp.get("output_text") or "")
        else:
            content = str(pdf_resp)

        if not content or len(content.strip()) < 20:
            raise HTTPException(status_code=500, detail="벡터스토어 내용 부족")
        
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

