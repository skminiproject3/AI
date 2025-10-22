import os
import re
import hashlib
import json
from typing import List, Tuple, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
import logging
from typing import List, Optional
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 업로드 폴더 정의
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # 폴더가 없으면 생성

# ====== Pydantic 모델 정의 ======
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

class QuizAnswer(BaseModel):
    question_text: str  # 질문 원문
    user_answer: str    # 사용자 답변

class QuizGradingRequest(BaseModel):
    pdf_paths: List[str]
    answers: List[QuizAnswer]
    # score_per_question: int = Field(20) # 백엔드에서 처리하도록 제거

# ====== 환경 설정 ======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Tavily 도구 초기화 (API Key가 없으면 None)
tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3) if TAVILY_API_KEY else None

app = FastAPI(title="PDF 학습 도우미 API")

# ===============================================
# ====== Core Functions (PDF/Vector/Text Processing) ======
# ===============================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF 파일에서 텍스트를 추출합니다."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return "\n".join([p.page_content for p in pages])
    except Exception as e:
        logger.error(f"PDF 텍스트 추출 실패 ({pdf_path}): {e}")
        return ""

def get_or_create_vectorstore(pdf_path: str) -> Optional[FAISS]:
    """PDF 경로를 기반으로 벡터스토어를 로드하거나 생성합니다."""
    base_vector_dir = "vector_stores"
    normalized_path = pdf_path.replace('\\', '/')
    path_hash = hashlib.md5(normalized_path.encode()).hexdigest()
    vector_path = os.path.join(base_vector_dir, path_hash)

    if not os.path.exists(base_vector_dir):
        os.makedirs(base_vector_dir)

    if os.path.exists(vector_path):
        return FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    else:
        logger.info(f"🔄 벡터스토어 생성 시작: {pdf_path}")
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            texts = [p.page_content for p in pages]
            if not texts:
                logger.warning(f"경고: {pdf_path}에서 텍스트를 추출하지 못했습니다.")
                return None
            vectorstore = FAISS.from_texts(texts, embeddings)
            vectorstore.save_local(vector_path)
            logger.info(f"✅ 벡터스토어 생성 및 저장 완료: {pdf_path}")
            return vectorstore
        except Exception as e:
            logger.error(f"❌ 벡터 생성 중 오류 발생 ({pdf_path}): {e}")
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


def split_by_subchapter(text: str) -> Dict[str, str]:
    """텍스트에서 '숫자.숫자' 형태(예: 2.1)를 기준으로 분리합니다."""
    pattern = r"(?=(\d+\.\d+))"
    splits = re.split(pattern, text)
    chapters = {}
    current_chapter = None
    for seg in splits:
        if re.match(r"\d+\.\d+", seg.strip()):
            current_chapter = seg.strip()
            chapters[current_chapter] = ""
        elif current_chapter:
            chapters[current_chapter] += seg.strip() + "\n"
    return chapters

# ===============================================
# ====== LLM Prompts and Functions (Summary & Quiz) ======
# ===============================================

summary_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
당신은 교재의 핵심 내용을 요약하는 전문가입니다. 다음 내용을 핵심 개념 중심으로 간결하고 명료하게 요약하세요.

- 주요 개념 3~5개 중심
- 공식, 정의, 특징 포함

내용:
{content}

출력 형식:
---
[요약]
1. ...
2. ...
3. ...
---
"""
)

quiz_generation_prompt = PromptTemplate(
    input_variables=["content", "num_questions", "difficulty"],
    template="""
다음 교재 내용을 바탕으로 {num_questions}개의 객관식 4지선다 문제를 생성하세요.
출력 형식은 **JSON 배열**로 반드시 다음 구조를 지켜야 합니다.

[
  {{
    "question": "문제 내용",
    "options": {{
      "a": "보기 a",
      "b": "보기 b",
      "c": "보기 c",
      "d": "보기 d"
    }}
  }},
  ...
]

- 난이도는 {difficulty} 입니다.
- 각 문제는 하나의 명확한 정답을 가져야 합니다.
- 정답 표시나 '정답:' 등은 절대 포함하지 마세요.
- 중복 문제 금지

내용:
{content}
"""
)

grading_prompt = PromptTemplate(
    input_variables=["question", "user_answer", "context"],
    template="""
당신은 주어진 '교재 내용'에만 근거하여 퀴즈를 채점하는 AI 조교입니다. 외부 지식을 절대 사용하지 마세요.

**[채점 절차]**
1. **정답 찾기**: '교재 내용'에서 '문제'에 대한 명확한 정답을 찾습니다.
2. **답변 비교**: '사용자 답변'을 정답과 비교합니다.
   - 객관식 문제: **사용자 답변이 'b) 키 교환' 형태이거나 단순히 'b' 또는 'b)'와 같이 정답 선택지 번호만 포함하는 경우 모두 정답으로 간주합니다.**
3. **출력 형식 준수**: 채점 결과를 아래 '출력 형식'에 맞춰 정확하게 작성합니다.

---
**교재 내용 (Source of Truth):**
{context}
---
**문제:**
{question}
---
**사용자 답변:**
{user_answer}
---
**[채점 결과 출력]**
정답여부: [정답/오답]
정답: [교재 내용에 근거한 명확하고 간결한 정답을 여기에 서술 (예: b) 키 교환)]
설명: [사용자 답변이 정답인 이유, 또는 오답인 경우 정확한 설명]
"""
)

def summarize_pdf_content(content: str) -> str:
    """전체 내용을 요약합니다."""
    prompt = summary_prompt.format(content=content[:10000]) # 4000자에서 10000자로 확장
    result = llm.invoke(prompt)
    return result.content.strip()

def summarize_subchapters(content: str, request_chapter: Optional[str] = None) -> Tuple[List[Dict[str, str]], str]:
    """단원별 요약을 처리합니다."""
    chapters = split_by_subchapter(content)
    if not chapters:
        return [], "❌ 문서에서 소단원(예: 4.1)을 찾을 수 없습니다."

    summaries = []
    
    # 1. 특정 단원 요청 (예: '4.1장' 또는 '4장')
    if request_chapter:
        req = request_chapter.strip().lower()
        matched_keys = []
        
        # 숫자.숫자 패턴 우선
        m_dot = re.search(r"(\d+\.\d+)", req)
        if m_dot:
            key = m_dot.group(1)
            if key in chapters:
                matched_keys.append(key)
        
        # 숫자 패턴 (상위 장)
        m_major = re.search(r"(\d+)", req)
        if not matched_keys and m_major:
            major = m_major.group(1) 
            matched_keys = sorted([k for k in chapters.keys() if k.startswith(f"{major}.")])
            
        if not matched_keys:
            return [], f"❌ 요청한 단원 '{request_chapter}' 에 해당하는 내용을 찾을 수 없습니다."
        
        # 매칭된 단원들만 결합 및 요약
        combined = "\n\n".join([chapters[k] for k in matched_keys])
        summary = summarize_pdf_content(combined)
        summaries.append({"chapter": ", ".join(matched_keys), "summaryText": summary})
        
        return summaries, "✅ 요청된 단원 요약 완료"

    # 2. 전체 단원 요약 요청 (request_chapter가 None일 때)
    for key, text in chapters.items():
        summary = summarize_pdf_content(text)
        summaries.append({"chapter": key, "summaryText": summary})
        
    return summaries, "✅ 단원별 전체 요약 완료"


# ===============================================
# ====== FastAPI Endpoints ======
# ===============================================

@app.get("/health/")
async def health_check():
    return {"status": "ok"}

@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    saved_files = []
    for file in files:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        try:
            with open(save_path, "wb") as f:
                f.write(await file.read())
            saved_files.append(save_path)
        except Exception as e:
            return {"error": str(e)}
    return {"saved_files": saved_files}

@app.post("/summarize/full")
async def summarize_full(request: PdfPathsRequest):
    _, content = combine_vectorstores(request.pdf_paths)
    if not content:
        raise HTTPException(status_code=400, detail="PDF 내용 불러오기 실패")
    prompt = f"다음 내용을 핵심 요약: {content[:10000]}"
    result = llm.invoke(prompt)
    return {"summaryText": result.content.strip()}

@app.post("/summarize/chapter")
async def summarize_chapter(request: ChapterRequest):
    _, content = combine_vectorstores(request.pdf_paths)
    
    if not content:
        raise HTTPException(status_code=400, detail="PDF 내용을 로드하지 못했습니다. 경로를 확인하세요.")
    
    summaries, message = summarize_subchapters(content, request_chapter=request.chapter_request)
    
    if not summaries:
         raise HTTPException(status_code=404, detail=message)
         
    return {"message": message, "summaries": summaries}

# -----------------------------------------------
# 질문 답변 엔드포인트
# -----------------------------------------------

@app.post("/question/")
async def question_endpoint(request: QuestionRequest):
    """질문 답변 (RAG + Web Search Fallback)."""
    question = request.question
    vectorstore, _ = combine_vectorstores(request.pdf_paths)
    
    if not vectorstore:
        raise HTTPException(status_code=400, detail="PDF 벡터 로드 실패. 경로를 확인하세요.")
        
    # 1. RAG 기반 유사도 검색
    docs = vectorstore.similarity_search(question, k=4)
    vector_context = "\n".join([d.page_content for d in docs])
    
    web_context = ""
    
    # 2. 벡터 컨텍스트 부족 시 (문서 1개 미만), Tavily 웹 검색 시도
    if not vector_context or len(vector_context.strip()) < 100:
        logger.info("📡 PDF에서 충분한 답을 찾지 못함 → 웹 검색 수행")
        try:
            if tavily_tool:
                tavily_results = tavily_tool.run(question)
                if tavily_results:
                    web_context += "\n\n--- 웹 검색 결과 (Tavily) ---\n\n"
                    for item in tavily_results:
                        web_context += f"출처: {item.get('url','N/A')}\n내용: {item.get('content', item.get('snippet',''))}\n\n"
            else:
                logger.warning("⚠️ Tavily API Key 없음 → DuckDuckGo 대체 검색")
                duck = DuckDuckGoSearchRun()
                search_text = duck.run(question)
                web_context += f"\n\n--- 웹 검색 결과 (DuckDuckGo) ---\n\n{search_text}\n"
        except Exception as e:
            logger.error(f"웹 검색 실패: {e}")
            web_context = ""

    # 3. 답변 프롬프트 구성
    final_context = f"--- 교재(PDF) 내용 ---\n{vector_context}\n{web_context}"
    
    prompt = f"""당신은 사용자가 제공한 교재 내용과 웹 검색 결과를 기반으로 질문에 답변하는 도우미입니다. 

    - 교재 내용(PDF)이 있다면 우선적으로 활용하여 답변을 작성하세요.
    - 교재 내용만으로 답변이 어렵거나, 교재 내용이 없는 경우에만 웹 검색 결과를 활용하여 답변하세요.
    - 웹 검색 결과를 활용했을 경우, 답변 말미에 '웹 검색 결과가 참고되었습니다.'라고 명시하세요.
    
    제공된 모든 컨텍스트:
    {final_context}
    
    질문: {question}
    
    답변:"""
    
    # 4. LLM 호출
    result = llm.invoke(prompt)
    answer = result.content.strip()
    return {"answer": answer}

# -----------------------------------------------
# 연습문제 엔드포인트
# -----------------------------------------------

@app.post("/quiz/generate")
async def quiz_generate(request: QuizGenerationRequest):
    _, content = combine_vectorstores(request.pdf_paths)
    if not content:
        raise HTTPException(status_code=400, detail="PDF 내용 불러오기 실패")
    prompt = f"다음 내용을 바탕으로 {request.num_questions}개 객관식 문제 생성, 난이도 {request.difficulty}: {content[:10000]}"
    result = llm.invoke(prompt)
    try:
        questions = json.loads(result.content.strip())
    except:
        questions = [{"question": result.content.strip()}]
    return {"questions": questions}

@app.post("/quiz/grade")
async def quiz_grade(request: QuizGradingRequest):
    """사용자의 답변을 채점하고 결과와 설명을 반환합니다."""
    _, content = combine_vectorstores(request.pdf_paths)
    
    if not content:
        raise HTTPException(status_code=400, detail="PDF 내용을 로드하지 못했습니다. 경로를 확인하세요.")

    results = []
    total_score = 0
    correct_count = 0
    num_total_questions = len(request.answers)
    score_per_question = 100 // num_total_questions if num_total_questions > 0 else 0
    
    for ans in request.answers:
        grade_prompt_text = grading_prompt.format(
            question=ans.question_text,
            user_answer=ans.user_answer,
            context=content[:10000]
        )

        try:
            result = llm.invoke(grade_prompt_text)
            result_text = result.content.strip()

            # LLM 응답에서 데이터 추출
            is_correct_str_match = re.search(r"정답여부:\s*([^\n]+)", result_text)
            correct_answer_match = re.search(r"정답:\s*([^\n]+)", result_text, re.DOTALL)
            explanation_match = re.search(r"설명:\s*([^\n]+)", result_text, re.DOTALL)

            is_correct = "정답" in is_correct_str_match.group(1).strip() if is_correct_str_match else False
            correct_answer = correct_answer_match.group(1).strip() if correct_answer_match else "채점 오류: 정답 미포함"
            explanation = explanation_match.group(1).strip() if explanation_match else "채점 오류: 설명 미포함"

            score = score_per_question if is_correct else 0
            total_score += score
            if is_correct:
                correct_count += 1
            
            results.append({
                "question": ans.question_text,
                "user_answer": ans.user_answer,
                "correct_answer": correct_answer,
                "explanation": explanation,
                "is_correct": is_correct,
                "score": score
            })

        except Exception as e:
            logger.error(f"채점 중 오류 발생: {e}")
            results.append({
                "question": ans.question_text,
                "user_answer": ans.user_answer,
                "correct_answer": "시스템 오류",
                "explanation": f"채점 중 오류 발생: {e}",
                "is_correct": False,
                "score": 0
            })
            
    # 최종 점수 보정 (100점 만점으로)
    final_total_score = total_score
    remaining_score = 100 - (score_per_question * num_total_questions)
    
    if remaining_score > 0 and correct_count > 0:
        for res in reversed(results):
            if res['is_correct']:
                res['score'] += remaining_score
                final_total_score += remaining_score
                break
    
    if correct_count == 0:
        final_total_score = 0

    return {
        "final_total_score": final_total_score,
        "correct_count": correct_count,
        "total_questions": num_total_questions,
        "results": results
    }