import os
import re
import hashlib
import random
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# ====== 환경 변수 로드 ======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY가 설정되지 않았습니다.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)

# ====== FastAPI 앱 ======
app = FastAPI(title="PDF 학습 도우미 API")

# ====== PDF/벡터 저장 경로 ======
PDF_DIR = "uploads"
VECTOR_DIR = "vector_stores"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ====== 벡터스토어 관리 (파일별) ======
vectorstore_registry = {}  # {pdf_hash: FAISS 객체}

# ====== 유틸 함수 ======
def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join([p.page_content for p in pages])

def get_or_create_vectorstore(pdf_path):
    path_hash = hashlib.md5(pdf_path.encode()).hexdigest()
    vector_path = os.path.join(VECTOR_DIR, path_hash)
    
    if vector_path in vectorstore_registry:
        return vectorstore_registry[vector_path]

    if os.path.exists(vector_path):
        vs = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore_registry[vector_path] = vs
        return vs
    else:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        texts = [p.page_content for p in pages]
        if not texts:
            return None
        vs = FAISS.from_texts(texts, embeddings)
        vs.save_local(vector_path)
        vectorstore_registry[vector_path] = vs
        return vs

def combine_vectorstores(pdf_paths: List[str]):
    vectorstores = []
    combined_content = ""
    for pdf_path in pdf_paths:
        vs = get_or_create_vectorstore(pdf_path)
        if vs:
            vectorstores.append(vs)
        content = extract_text_from_pdf(pdf_path)
        combined_content += content + "\n\n--- PDF 분리 마커 ---\n\n"
    
    if not vectorstores:
        return None, None
    
    main_vs = vectorstores[0]
    for vs in vectorstores[1:]:
        main_vs.merge_from(vs)
    
    return main_vs, combined_content

def split_by_subchapter(text):
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

# ====== 프롬프트 정의 ======
summary_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
다음은 교재의 내용입니다. 핵심 개념 중심으로 요약을 작성하세요.

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

quiz_prompt = PromptTemplate(
    input_variables=["context", "variation"],
    template="""
다음 교재 내용을 바탕으로 5문항을 출제하세요. **(참고: 내용은 여러 PDF 파일에서 결합된 것일 수 있습니다.)**

- {variation} 유형으로 작성
- 객관식, 단답형, 서술형을 섞기
- 문제 중복 금지

내용:
{context}

출력 형식:
---
[연습문제]
1. ...
2. ...
3. ...
4. ...
5. ...
---
"""
)

grading_prompt = PromptTemplate(
    input_variables=["question", "user_answer", "context"],
    template="""
당신은 주어진 '교재 내용'에만 근거하여 퀴즈를 채점하는 AI 조교입니다. 외부 지식을 절대 사용하지 마세요.

**[채점 절차]**
1. 문제 분석
2. 정답 찾기
3. 답변 비교
4. 점수 부여: 맞으면 20점, 틀리면 0점

---
교재 내용:
{context}
---
문제:
{question}
---
사용자 답변:
{user_answer}
---
출력:
정답여부: [정답/오답]
정답: [교재 내용 기반]
점수: [20 또는 0]
"""
)

# ====== 요약 ======
def summarize_pdf(content: str):
    prompt = summary_prompt.format(content=content[:4000])
    result = llm.invoke(prompt)
    return result.content.strip()

def summarize_requested_subchapter(content: str, user_request: str):
    chapters = split_by_subchapter(content)
    m = re.search(r"(\d+\.\d+)", user_request)
    if m and m.group(1) in chapters:
        text = chapters[m.group(1)]
        prompt = summary_prompt.format(content=text[:4000])
        result = llm.invoke(prompt)
        return [m.group(1)], result.content.strip()
    else:
        # 전체 단원 합쳐서 요약
        combined = "\n".join(chapters.values())
        prompt = summary_prompt.format(content=combined[:4000])
        result = llm.invoke(prompt)
        return list(chapters.keys()), result.content.strip()

# ====== 퀴즈 생성 ======
def generate_quiz(content: str, num_questions: int =5):
    variation = random.choice(["핵심 개념형", "응용형", "이해형", "추론형"])
    prompt = quiz_prompt.format(context=content[:4000], variation=variation)
    result = llm.invoke(prompt)
    return result.content.strip()

# ====== 질문 답변 ======
def answer_question_with_vector(vectorstore, question: str):
    if not vectorstore:
        return "❌ 로드된 문서가 없어 답변할 수 없습니다."
    
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""
다음 교재 내용을 기반으로 질문에 답하세요.
교재에 없으면 '교재에 관련 내용이 없습니다.'라고 답하세요.

교재 내용:
{context}

질문:
{question}

답변:
"""
    result = llm.invoke(prompt)
    answer = result.content.strip()

    if "교재에 관련 내용이 없습니다" in answer or len(answer) < 30:
        try:
            search_results = tavily_tool.invoke({"query": question})
            combined_context = "\n".join([r["content"] for r in search_results])
            web_prompt = f"""
웹 검색 결과를 바탕으로 상세하게 답변해주세요.

검색 결과:
{combined_context}

질문:
{question}

답변:
"""
            web_answer = llm.invoke(web_prompt)
            return web_answer.content.strip()
        except Exception as e:
            return f"웹 검색 중 오류: {e}"
    return answer

# ====== API 엔드포인트 ======

@app.post("/upload_pdfs")
async def upload_pdfs(files: List[UploadFile]):
    pdf_paths = []
    for file in files:
        path = os.path.join(PDF_DIR, file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())
        pdf_paths.append(path)
    
    vectorstore, content = combine_vectorstores(pdf_paths)
    if not vectorstore:
        return JSONResponse(status_code=400, content={"error": "PDF 처리 실패"})
    
    return {"message": "PDF 처리 완료", "content_length": len(content)}

@app.post("/summary")
def api_summary(content: str = Form(...), subchapter: Optional[str] = None):
    if subchapter:
        keys, summary = summarize_requested_subchapter(content, subchapter)
        return {"subchapter": keys, "summary": summary}
    else:
        summary = summarize_pdf(content)
        return {"summary": summary}

@app.post("/quiz")
def api_quiz(content: str = Form(...)):
    quiz_text = generate_quiz(content)
    return {"quiz": quiz_text}

@app.post("/ask")
def api_ask(content: str = Form(...), question: str = Form(...)):
    # 현재는 하나의 vectorstore만 사용 (통합 content 기반)
    vectorstore, _ = combine_vectorstores([])  # 실제 API에서는 저장된 vectorstore를 사용
    answer = answer_question_with_vector(vectorstore, question)
    return {"answer": answer}