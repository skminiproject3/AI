import os
import re
import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
import hashlib

# ====== 환경 설정 ======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY) # 온도를 낮춰 일관성 확보
tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# ====== PDF 로드 ======
def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join([p.page_content for p in pages])

# ====== 벡터스토어 생성/로드 및 병합 (개선) ======
def get_or_create_vectorstore(pdf_path):
    # 단일 PDF에 대한 벡터스토어 생성/로드
    base_vector_dir = "vector_stores"
    path_hash = hashlib.md5(pdf_path.encode()).hexdigest()
    vector_path = os.path.join(base_vector_dir, path_hash)

    if not os.path.exists(base_vector_dir):
        os.makedirs(base_vector_dir)

    if os.path.exists(vector_path):
        # print(f"✅ 기존 벡터DB 로드 중... ({pdf_path})") # 너무 길어 생략
        return FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"🧠 벡터 생성 중... ({pdf_path})")
        loader = PyPDFLoader(pdf_path)
        try:
            pages = loader.load()
            texts = [p.page_content for p in pages]
            if not texts:
                print(f"❌ PDF에서 텍스트를 추출하지 못했습니다: {pdf_path}")
                return None
            vectorstore = FAISS.from_texts(texts, embeddings)
            vectorstore.save_local(vector_path)
            # print("✅ 벡터 생성 및 저장 완료!") # 너무 길어 생략
            return vectorstore
        except Exception as e:
            print(f"❌ 벡터 생성 중 오류 발생 ({pdf_path}): {e}")
            return None

def combine_vectorstores(pdf_paths):
    """여러 PDF 경로로부터 벡터스토어를 생성/로드하고 하나로 병합합니다."""
    vectorstores = []
    combined_content = ""
    
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"⚠️ 파일이 존재하지 않습니다. 건너뜁니다: {pdf_path}")
            continue

        # 1. 벡터스토어 처리
        vs = get_or_create_vectorstore(pdf_path)
        if vs:
            vectorstores.append(vs)
            print(f"✅ {pdf_path} 벡터 준비 완료.")

        # 2. 텍스트 내용 결합
        content = extract_text_from_pdf(pdf_path)
        combined_content += content + "\n\n--- PDF 분리 마커 ---\n\n"

    if not vectorstores:
        print("❌ 유효한 PDF 파일이 없어 벡터스토어를 병합할 수 없습니다.")
        return None, None
    
    if len(vectorstores) == 1:
        print("✅ 단일 벡터스토어 로드/생성 완료.")
        return vectorstores[0], combined_content
    
    # 3. 다수의 벡터스토어 병합
    print(f"🔄 {len(vectorstores)}개의 벡터스토어를 하나로 병합 중...")
    
    # 첫 번째 벡터스토어를 기준으로 나머지 벡터스토어를 병합
    main_vectorstore = vectorstores[0]
    for i in range(1, len(vectorstores)):
        main_vectorstore.merge_from(vectorstores[i])
    
    print("✅ 모든 벡터스토어 병합 완료!")
    return main_vectorstore, combined_content


# ====== 단원(2.1, 2.2 등) 분리 ======
# (이 함수는 결합된 텍스트에서 단원을 분리하는 데 사용되며, 로직 변경 없음)
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

# ====== 프롬프트 (변경 없음) ======
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
- 각 문항 끝에 '(정답: ...)' 은 포함하지 말 것

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

# 채점 프롬프트: 객관식 답변 유연성 유지
grading_prompt = PromptTemplate(
input_variables=["question", "user_answer", "context"],
    template="""
당신은 주어진 '교재 내용'에만 근거하여 퀴즈를 채점하는 AI 조교입니다. 외부 지식을 절대 사용하지 마세요.

**[채점 절차]**
1. **문제 분석**: 주어진 '문제'를 정확히 이해합니다.
2. **정답 찾기**: '교재 내용'에서 문제에 대한 명확한 정답을 찾습니다. 이것이 유일한 채점 기준입니다.
3. **답변 비교**: '사용자 답변'을 위에서 찾은 정답과 비교합니다.
    - 객관식 문제: **사용자 답변이 'b) 키 교환' 형태이거나 단순히 'b'와 같이 정답 선택지 번호만 포함하는 경우 모두 정답으로 간주합니다.**
    - 서술형 문제: 사용자의 답변이 교재 내용의 핵심을 포함하는지 확인합니다.
4. **점수 부여**:
    - 사용자의 답변이 옳다면 20점을 부여합니다.
    - 사용자의 답변이 틀리다면 0점을 부여합니다.
5. **출력 형식 준수**: 채점 결과를 아래 '출력 형식'에 맞춰 정확하게 작성합니다.

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
정답: [교재 내용에 근거한 명확하고 간결한 정답을 여기에 서술]
점수: [20 또는 0]
"""
)

# ====== 요약, 퀴즈, 세션, 답변 함수 (로직 변경 없음, 입력이 combined_content와 vectorstore로 변경) ======
def summarize_pdf(content):
    prompt = summary_prompt.format(content=content[:4000])
    result = llm.invoke(prompt)
    return result.content.strip()

def summarize_by_subchapter(content):
    chapters = split_by_subchapter(content)
    summaries = {}
    for key, text in chapters.items():
        # PDF 분리 마커는 단원 분리에 영향을 미칠 수 있으나,
        # 이 함수는 2.1, 2.2 형태의 패턴에 의존하므로 텍스트 길이에 맞춰 처리.
        prompt = summary_prompt.format(content=text[:4000]) 
        result = llm.invoke(prompt)
        summaries[key] = result.content.strip()
    return summaries

def generate_quiz(content):
    variation = random.choice(["핵심 개념형", "응용형", "요약 이해형", "추론형", "비판적 사고형"])
    prompt = quiz_prompt.format(context=content[:4000], variation=variation)
    result = llm.invoke(prompt)
    return result.content.strip()

def run_quiz_session(content):
 # 문제 생성
    quiz_text = generate_quiz(content)
    print("\n--- 연습문제 ---")
    
    quiz_body = quiz_text.split('[연습문제]')[-1].strip()
    
    # [개선] 정규 표현식을 사용하여 문항 번호와 문항 내용(선택지 포함)을 정확히 분리
    # \d+\.\s* : 숫자와 점, 공백으로 시작
    # (.*?) : 문항 내용 (최소 매칭)
    # (?=\n\d+\.|\Z) : 다음 문항 번호(\n\d+\.) 또는 문자열의 끝(\Z)까지 매칭
    questions_raw = re.findall(r"\d+\.\s*(.*?)(?=\n\d+\.|\Z)", quiz_body, re.DOTALL)
    
    # 출력 시에는 [연습문제] 헤더와 추출된 문항 내용 전체를 보여줍니다.
    print(quiz_body)

    if not questions_raw:
        print("❌ 문제를 파싱할 수 없습니다. 생성된 문제 형식을 확인해주세요.")
        return

    print("\n🧩 퀴즈를 시작합니다! (각 문항은 20점)\n")

    results = []
    total_score = 0
    correct_count = 0

    for idx, q_text in enumerate(questions_raw, 1):
        # 문항 번호를 붙여 다시 출력 (사용자에게 현재 진행 중인 문제임을 명확히 알림)
        # re.sub를 사용하여 문항 내용 시작 부분의 공백과 개행문자 정리
        q_clean = re.sub(r'^\s+|\s+$', '', q_text)
        
        print(f"\n문항 {idx}: {q_clean}")
        
        # [핵심] 사용자 입력 대기
        user_answer = input("당신의 답: ").strip()

        # 채점 프롬프트 구성 및 LLM 호출
        grade_prompt_text = grading_prompt.format(
            question=q_clean, # 문항 내용 전체 (선택지 포함)
            user_answer=user_answer,
            context=content[:4000] 
        )
        
        # ... (이하 채점 로직은 이전 코드와 동일) ...
        try:
            result = llm.invoke(grade_prompt_text)
            result_text = result.content.strip()

            is_correct_str = (re.search(r"정답여부:\s*([^\n]+)", result_text).group(1).strip() if re.search(r"정답여부:\s*([^\n]+)", result_text) else "오답")
            is_correct = "정답" in is_correct_str

            correct_answer_match = re.search(r"정답:\s*([^\n]+)", result_text, re.DOTALL)
            correct_answer = correct_answer_match.group(1).strip() if correct_answer_match else "채점 오류: 정답을 찾을 수 없음"

            score_match = re.search(r"점수:\s*(\d+)", result_text)
            score = int(score_match.group(1)) if score_match else (20 if is_correct else 0)

            if score >= 10: 
                correct_count += 1
                is_correct = True
            
            total_score += score
            
            results.append({
                "question": q_clean,
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "score": score
            })

        except Exception as e:
            print(f"❌ 문항 {idx} 채점 중 오류 발생: {e}")
            results.append({
                "question": q_clean,
                "user_answer": user_answer,
                "correct_answer": "시스템 오류로 채점 불가",
                "is_correct": False,
                "score": 0
            })
            continue

    # 최종 결과 리포트 출력
    print("\n\n--- 연습문제 결과 ---")
    for idx, res in enumerate(results, 1):
        print("-" * 40)
        print(f"문항 {idx}: {res['question']}")
        print(f"당신의 답: {res['user_answer']}")
        print(f"정답: {res['correct_answer']}")
        result_symbol = "✅ (정답)" if res['is_correct'] else "❌ (틀림)"
        print(f"결과: {result_symbol}")
    
    print("-" * 40)
    print(f"\n=== 🏁 최종 결과 ===")
    print(f"총 {correct_count}문제 정답 / {len(questions_raw)}문제 ({total_score}점)")
    print("-" * 40)

def answer_question_with_vector(vectorstore, question):
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
        print("\n📡 교재에 충분한 정보가 없어 웹 검색을 함께 활용합니다...\n")
        try:
            search_results = tavily_tool.invoke({"query": question})
            combined_context = "\n".join([r["content"] for r in search_results])
            web_prompt = f"""
            당신은 질문에 답변하는 AI 어시스턴트입니다.
            주어진 웹 검색 결과를 바탕으로 사용자의 질문에 대해 상세하고 친절하게 답변해주세요.

            웹 검색 결과:
            {combined_context}

            질문:
            {question}

            답변:
            """
            web_answer = llm.invoke(web_prompt)
            return web_answer.content.strip()
        except Exception as e:
            return f"웹 검색 중 오류가 발생했습니다: {e}"

    return answer

# ====== 메인 (개선) ======
def main():
    print("=== PDF 학습 도우미 (다중 파일 지원) ===")
    # 쉼표로 구분된 여러 파일 경로를 입력받음
    pdf_paths_input = input("PDF 파일 경로들을 쉼표(,)로 구분하여 입력하세요 (예: file1.pdf, file2.pdf): ").strip()
    
    if not pdf_paths_input:
        print("❌ 입력된 파일 경로가 없습니다.")
        return
        
    # 경로를 분리하고 공백 제거
    pdf_paths = [p.strip() for p in pdf_paths_input.split(',') if p.strip()]

    if not pdf_paths:
        print("❌ 유효한 파일 경로가 입력되지 않았습니다.")
        return

    # 다중 PDF 파일 처리 및 병합
    print("\n--- 문서 로드 및 벡터 처리 ---")
    vectorstore, content = combine_vectorstores(pdf_paths)

    if not vectorstore:
        print("❌ 프로그램을 실행할 수 없습니다. 유효한 PDF 파일을 확인하세요.")
        return

    print("\n✅ 모든 PDF 로드 및 벡터 병합 완료!\n")

    while True:
        print("\n선택하세요:")
        print("1. 연습문제 + 채점 모드 (통합 내용 기반)")
        print("2. 질문하기 (통합 내용 기반 + 웹 검색)")
        print("3. 전체 요약 (통합 내용 기반)")
        print("4. 단원별 요약 (통합 내용 기반)")
        print("0. 종료")
        choice = input(">>> ").strip()

        if choice == "1":
            try:
                run_quiz_session(content)
            except Exception as e:
                print(f"❌ 연습문제 세션 오류: {e}")

        elif choice == "2":
            question = input("질문을 입력하세요: ").strip()
            try:
                answer = answer_question_with_vector(vectorstore, question)
                print("\n[답변]")
                print(answer)
            except Exception as e:
                print(f"질문 처리 중 오류: {e}")

        elif choice == "3":
            try:
                print("\n--- 전체 요약 ---")
                summary = summarize_pdf(content)
                print(summary)
            except Exception as e:
                print(f"PDF 요약 중 오류: {e}")

        elif choice == "4":
            try:
                print("\n--- 단원별 요약 ---")
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