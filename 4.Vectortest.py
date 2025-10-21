import os
import re
import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS

# ====== 환경 설정 ======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, api_key=OPENAI_API_KEY)
tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# ====== PDF 로드 ======
def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join([p.page_content for p in pages])

# ====== 벡터스토어 생성/로드 ======
def get_vectorstore(pdf_path):
    vector_path = "pdf_vectors"
    if os.path.exists(vector_path):
        print("✅ 기존 벡터DB 로드 중...")
        return FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("🧠 벡터 생성 중...")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        texts = [p.page_content for p in pages]
        vectorstore = FAISS.from_texts(texts, embeddings)
        vectorstore.save_local(vector_path)
        print("✅ 벡터 생성 및 저장 완료!")
        return vectorstore

# ====== 단원(2.1, 2.2 등) 분리 ======
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

# ====== 프롬프트 ======
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
다음 교재 내용을 바탕으로 5문항을 출제하세요.

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

question_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
다음 교재 내용을 기반으로 질문에 답하세요.
교재에 없으면 '외부 지식 필요'라고 답하세요.

교재 내용:
{context}

질문:
{question}

출력:
"""
)

# ====== 요약 ======
def summarize_pdf(content):
    prompt = summary_prompt.format(content=content[:3000])
    result = llm.invoke(prompt)
    return result.content.strip()

# ====== 단원별 요약 ======
def summarize_by_subchapter(content):
    chapters = split_by_subchapter(content)
    summaries = {}
    for key, text in chapters.items():
        prompt = summary_prompt.format(content=text[:2500])
        result = llm.invoke(prompt)
        summaries[key] = result.content.strip()
    return summaries

# ====== 연습문제 생성 ======
def generate_quiz(content):
    variation = random.choice(["핵심 개념형", "응용형", "요약 이해형", "추론형", "비판적 사고형"])
    prompt = quiz_prompt.format(context=content[:2500], variation=variation)
    result = llm.invoke(prompt)
    return result.content.strip()

# ====== 벡터 기반 질문 ======
def answer_question_with_vector(vectorstore, question):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])

    prompt = question_prompt.format(context=context, question=question)
    result = llm.invoke(prompt)
    answer = result.content.strip()

    if "외부 지식 필요" in answer or len(answer) < 25:
        print("\n📡 교재에 답이 없어 웹 검색 중...\n")
        search_results = tavily_tool.invoke({"query": question})
        combined = "\n".join([r["content"] for r in search_results])
        web_answer = llm.invoke(f"검색 결과 기반으로 답하세요:\n{combined}\n\n질문:{question}")
        return web_answer.content.strip()

    return answer

# ====== 메인 ======
def main():
    print("=== PDF 학습 도우미 ===")
    pdf_path = input("PDF 파일 경로를 입력하세요: ").strip()

    if not os.path.exists(pdf_path):
        print("❌ 파일이 존재하지 않습니다.")
        return

    content = extract_text_from_pdf(pdf_path)
    vectorstore = get_vectorstore(pdf_path)

    print("\n✅ PDF 로드 및 벡터 준비 완료!\n")

    while True:
        print("\n선택하세요:")
        print("1. 연습문제 생성")
        print("2. 질문하기 (빠른 벡터 검색 기반)")
        print("3. 전체 요약")
        print("4. 단원별 요약")
        print("0. 종료")
        choice = input(">>> ").strip()

        if choice == "1":
            print("\n--- 연습문제 ---")
            try:
                quiz = generate_quiz(content)
                print(quiz)
            except Exception as e:
                print(f"연습문제 생성 오류: {e}")

        elif choice == "2":
            question = input("질문을 입력하세요: ").strip()
            try:
                answer = answer_question_with_vector(vectorstore, question)
                print("\n답변:", answer)
            except Exception as e:
                print(f"질문 처리 중 오류: {e}")

        elif choice == "3":
            print("\n--- 전체 요약 ---")
            try:
                summary = summarize_pdf(content)
                print(summary)
            except Exception as e:
                print(f"PDF 요약 중 오류: {e}")

        elif choice == "4":
            print("\n--- 단원별 요약 ---")
            try:
                summaries = summarize_by_subchapter(content)
                for ch, summ in summaries.items():
                    print(f"\n[{ch} 단원 요약]\n{summ}\n")
            except Exception as e:
                print(f"단원별 요약 중 오류: {e}")

        elif choice == "0":
            print("프로그램을 종료합니다.")
            break

        else:
            print("❌ 잘못된 선택입니다. 다시 입력하세요.")

if __name__ == "__main__":
    main()
