# main.py
import os
import json
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일 확인 필요")

# LangChain 패키지
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Gradio
import gradio as gr
from gradio_pdf import PDF

# 전역 변수
current_vectorstore = None
current_pdf_path = None
current_questions = []
current_answers = []

# ----------------------------
# PDF 로딩 및 벡터 저장소 생성
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
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        separators=["\n\n", "\n", ".", " ", ""]
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
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        template = "다음 문서를 요약해주세요:\n{context}\n요약:"
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
        doc_chain = create_stuff_documents_chain(model, prompt)
        chain = create_retrieval_chain(retriever, doc_chain)
        result = chain.invoke({"input": "요약"})
        return result['answer']
    except Exception as e:
        return f"PDF 요약 중 오류: {str(e)}"

# ----------------------------
# 연습문제 생성 (문제/정답 분리)
# ----------------------------
def generate_exercises(pdf_file):
    global current_questions, current_answers
    try:
        vectorstore = load_pdf_to_vector_store(pdf_file)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        template = '''
다음 문서를 참고하여 3개의 연습문제를 만들고 각 문제의 정답을 JSON 형태로 반환해주세요.
예시:
[
    {"question": "문제 1 내용", "answer": "정답 1"},
    {"question": "문제 2 내용", "answer": "정답 2"},
    {"question": "문제 3 내용", "answer": "정답 3"}
]

문서:
{context}
'''
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
        doc_chain = create_stuff_documents_chain(model, prompt)
        chain = create_retrieval_chain(retriever, doc_chain)
        result = chain.invoke({"input": "연습문제 생성"})

        try:
            exercises = json.loads(result['answer'])
        except:
            return "연습문제 생성 실패: JSON 변환 불가", []

        current_questions = [ex['question'] for ex in exercises]
        current_answers = [ex['answer'] for ex in exercises]
        return "\n\n".join(current_questions), ""  # 문제만 먼저 표시
    except Exception as e:
        return f"연습문제 생성 오류: {str(e)}", ""

# ----------------------------
# 정답 공개
# ----------------------------
def show_answer(index):
    if 0 <= index < len(current_answers):
        return current_answers[index]
    return "정답 없음"

# ----------------------------
# 질문 답변 (PDF 기반)
# ----------------------------
def answer_question(pdf_file, question):
    try:
        vectorstore = load_pdf_to_vector_store(pdf_file)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        template = '''다음 문맥을 바탕으로 질문에 정확하게 답변해주세요.
문맥에서 관련 정보를 찾을 수 없다면, "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답변해주세요.

<문맥>
{context}
</문맥>

질문: {input}

답변:'''
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, api_key=OPENAI_API_KEY)
        doc_chain = create_stuff_documents_chain(model, prompt)
        chain = create_retrieval_chain(retriever, doc_chain)
        response = chain.invoke({'input': question})
        return response['answer']
    except Exception as e:
        return f"질문 처리 중 오류: {str(e)}"

# ----------------------------
# Gradio 인터페이스
# ----------------------------
def create_interface():
    with gr.Blocks(title="학습 PDF QA & 요약 시스템") as demo:
        gr.Markdown("# 학습 PDF QA & 요약 시스템")
        gr.Markdown("PDF 업로드 → 요약, 연습문제 생성, 질문 답변 기능 제공")

        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = PDF(label="PDF 업로드")

                summarize_btn = gr.Button("📄 PDF 요약")
                exercise_btn = gr.Button("📝 연습문제 생성")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="💬 질문")
                msg = gr.Textbox(label="질문 입력", placeholder="PDF 내용에 대해 질문...")

                submit_btn = gr.Button("📤 질문하기")
                clear_btn = gr.Button("🗑️ 대화 초기화")

                # 연습문제 정답 공개
                answer_index = gr.Number(label="정답 공개 번호", value=0, precision=0)
                show_answer_btn = gr.Button("정답 확인")
                answer_output = gr.Textbox(label="정답")

        # PDF 요약 버튼
        summarize_btn.click(lambda pdf_file: [(pdf_file, summarize_pdf(pdf_file))], pdf_input, chatbot)
        # 연습문제 버튼
        exercise_btn.click(generate_exercises, pdf_input, [chatbot, answer_output])
        # 질문 버튼
        def respond(message, chat_history, pdf_file):
            if not message.strip():
                return chat_history, ""
            answer = answer_question(pdf_file, message)
            chat_history.append((message, answer))
            return chat_history, ""
        submit_btn.click(respond, [msg, chatbot, pdf_input], [chatbot, msg])
        msg.submit(respond, [msg, chatbot, pdf_input], [chatbot, msg])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

        # 정답 공개 버튼
        show_answer_btn.click(lambda i: show_answer(int(i)), answer_index, answer_output)

    return demo

# ----------------------------
# 실행
# ----------------------------
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        debug=True,
        server_name="127.0.0.1",
        server_port=7860
    )
