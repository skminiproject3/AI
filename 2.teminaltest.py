# main_terminal_v2.py
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage

# ----------------------------
# 환경 변수 로드
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(".env 파일에서 OPENAI_API_KEY 확인 필요")

# ----------------------------
# 전역 변수
# ----------------------------
current_vectorstore = None
current_pdf_path = None
current_questions = []
current_answers = []

# ----------------------------
# PDF 로딩 및 벡터 생성
# ----------------------------
def load_pdf_to_vector_store(pdf_file, chunk_size=1000, chunk_overlap=100):
    global current_vectorstore, current_pdf_path
    if current_vectorstore is not None and current_pdf_path == pdf_file:
        return current_vectorstore

    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    if not documents:
        raise ValueError("PDF에서 텍스트 추출 실패")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(splits, embeddings)

    current_vectorstore = vectorstore
    current_pdf_path = pdf_file
    return vectorstore

# ----------------------------
# PDF 요약
# ----------------------------
def summarize_pdf(pdf_file):
    try:
        vectorstore = load_pdf_to_vector_store(pdf_file)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        docs = retriever.get_relevant_documents("요약")
        context_text = "\n".join([d.page_content for d in docs])

        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
        human_msg = HumanMessage(content=f"다음 문서를 요약해주세요:\n{context_text}\n\n요약:")
        result = model([human_msg])
        return result.content

    except Exception as e:
        return f"PDF 요약 중 오류: {str(e)}"

# ----------------------------
# 연습문제 생성
# ----------------------------
def generate_exercises(pdf_file):
    global current_questions, current_answers
    try:
        vectorstore = load_pdf_to_vector_store(pdf_file)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents("연습문제 생성")
        context_text = "\n".join([d.page_content for d in docs])

        instruction = f"""
다음 문서를 참고하여 3개의 연습문제를 만들고 각 문제의 정답을 JSON 형태로 반환해주세요.
예시:
[
    {{"question": "문제 1 내용", "answer": "정답 1"}},
    {{"question": "문제 2 내용", "answer": "정답 2"}},
    {{"question": "문제 3 내용", "answer": "정답 3"}}
]
문서:
{context_text}
"""
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
        human_msg = HumanMessage(content=instruction)
        result = model([human_msg])

        exercises = json.loads(result.content)
        current_questions = [ex['question'] for ex in exercises]
        current_answers = [ex['answer'] for ex in exercises]
        return current_questions

    except Exception as e:
        return [f"연습문제 생성 오류: {str(e)}"]

# ----------------------------
# 질문 답변
# ----------------------------
def answer_question(pdf_file, question):
    try:
        vectorstore = load_pdf_to_vector_store(pdf_file)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(question)
        context_text = "\n".join([d.page_content for d in docs])

        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
        human_msg = HumanMessage(content=f"문맥:\n{context_text}\n\n질문: {question}\n답변:")
        result = model([human_msg])
        return result.content

    except Exception as e:
        return f"질문 처리 오류: {str(e)}"

# ----------------------------
# 정답 확인
# ----------------------------
def show_answer(index):
    if 0 <= index < len(current_answers):
        return current_answers[index]
    return "정답 없음"

# ----------------------------
# 터미널 인터페이스
# ----------------------------
def main():
    print("=== PDF 학습 QA & 요약 시스템 ===")
    pdf_path = input("분석할 PDF 경로를 입력하세요: ").strip()

    while True:
        print("\n선택하세요:")
        print("1. PDF 요약")
        print("2. 연습문제 생성")
        print("3. 질문하기")
        print("4. 정답 확인")
        print("0. 종료")
        choice = input(">>> ").strip()

        if choice == "1":
            print("\n--- PDF 요약 ---")
            print(summarize_pdf(pdf_path))

        elif choice == "2":
            print("\n--- 연습문제 ---")
            questions = generate_exercises(pdf_path)
            for idx, q in enumerate(questions):
                print(f"{idx+1}. {q}")

        elif choice == "3":
            question = input("질문 입력: ")
            print(f"답변: {answer_question(pdf_path, question)}")

        elif choice == "4":
            index = int(input("정답 번호 입력: ")) - 1
            print(f"정답: {show_answer(index)}")

        elif choice == "0":
            break

        else:
            print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()
