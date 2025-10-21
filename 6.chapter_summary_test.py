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

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# ====== PDF 로드 ======
def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join([p.page_content for p in pages])

# ====== 벡터스토어 생성/로드 및 병합 (개선) ======
def get_or_create_vectorstore(pdf_path):
    base_vector_dir = "vector_stores"
    path_hash = hashlib.md5(pdf_path.encode()).hexdigest()
    vector_path = os.path.join(base_vector_dir, path_hash)

    if not os.path.exists(base_vector_dir):
        os.makedirs(base_vector_dir)

    if os.path.exists(vector_path):
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
            return vectorstore
        except Exception as e:
            print(f"❌ 벡터 생성 중 오류 발생 ({pdf_path}): {e}")
            return None

def combine_vectorstores(pdf_paths):
    vectorstores = []
    combined_content = ""
    
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"⚠️ 파일이 존재하지 않습니다. 건너뜁니다: {pdf_path}")
            continue

        vs = get_or_create_vectorstore(pdf_path)
        if vs:
            vectorstores.append(vs)
            print(f"✅ {pdf_path} 벡터 준비 완료.")

        content = extract_text_from_pdf(pdf_path)
        combined_content += content + "\n\n--- PDF 분리 마커 ---\n\n"

    if not vectorstores:
        print("❌ 유효한 PDF 파일이 없어 벡터스토어를 병합할 수 없습니다.")
        return None, None
    
    if len(vectorstores) == 1:
        print("✅ 단일 벡터스토어 로드/생성 완료.")
        return vectorstores[0], combined_content
    
    print(f"🔄 {len(vectorstores)}개의 벡터스토어를 하나로 병합 중...")
    main_vectorstore = vectorstores[0]
    for i in range(1, len(vectorstores)):
        main_vectorstore.merge_from(vectorstores[i])
    print("✅ 모든 벡터스토어 병합 완료!")
    return main_vectorstore, combined_content

# ====== 단원(2.1, 2.2 등) 분리 ======
def split_by_subchapter(text):
    """
    텍스트에서 '숫자.숫자' 형태(예: 2.1, 4.1)를 기준으로 분리.
    반환: { '2.1': '내용...', '2.2': '내용...' }
    """
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

# ====== 요약 관련 함수 ======
def summarize_pdf(content):
    prompt = summary_prompt.format(content=content[:4000])
    result = llm.invoke(prompt)
    return result.content.strip()

def summarize_by_subchapter(content):
    """
    전체 단원(소단원)들을 분리해서 각각 요약합니다.
    반환: (chapters_dict, summaries_dict)
    """
    chapters = split_by_subchapter(content)
    summaries = {}
    for key, text in chapters.items():
        # 길이가 길면 잘라서 보내기
        prompt = summary_prompt.format(content=text[:4000])
        result = llm.invoke(prompt)
        summaries[key] = result.content.strip()
    return chapters, summaries

def summarize_requested_subchapter(content, user_request):
    """
    사용자가 '4.1장', '4.1', '4장', '4' 등으로 요청할 때, 가장 적절한 단원(들)을 찾아 요약을 반환.
    반환: (found_keys_list, summary_text)
    """
    chapters = split_by_subchapter(content)
    if not chapters:
        return [], "❌ 문서에서 소단원을 찾을 수 없습니다."

    # 정규화: '4.1장', '4장 요약 해줘' 등에서 숫자부분만 추출
    req = user_request.strip().lower()
    # 숫자.숫자 패턴 우선
    m = re.search(r"(\d+\.\d+)", req)
    if m:
        key = m.group(1)
        if key in chapters:
            text = chapters[key]
            prompt = summary_prompt.format(content=text[:4000])
            result = llm.invoke(prompt)
            return [key], result.content.strip()
        else:
            return [], f"❌ 요청한 소단원 '{key}' 을(를) 문서에서 찾지 못했습니다."

    # '4장' 또는 '4' 형태 (상위 장 요청) -> '4.'로 시작하는 모든 소단원 합쳐서 요약
    m2 = re.search(r"(\d+)\s*장?", req)
    if m2:
        major = m2.group(1)  # 예: '4'
        # 소단원 키 중 '4.'로 시작하는 것들 수집
        matched_keys = sorted([k for k in chapters.keys() if k.startswith(f"{major}.")])
        if not matched_keys:
            return [], f"❌ '{major}장'에 해당하는 소단원을 찾을 수 없습니다."
        combined = "\n\n".join([chapters[k] for k in matched_keys])
        prompt = summary_prompt.format(content=combined[:4000])
        result = llm.invoke(prompt)
        return matched_keys, result.content.strip()

    # 숫자 패턴이 전혀 없으면 시도: 사용자가 '4.1장 요약' 외 자유형으로 요청했을 경우
    # 가능한 가장 근접한 키(예: '4.1' 포함하는 것) 찾아 반환
    # 여기서는 숫자 포함 키를 탐색
    digits = re.findall(r"\d+", req)
    if digits:
        # 예: 첫번째 숫자와 점으로 매칭 시도
        for d in digits:
            sub_matches = [k for k in chapters.keys() if k.startswith(f"{d}.")]
            if sub_matches:
                combined = "\n\n".join([chapters[k] for k in sub_matches])
                prompt = summary_prompt.format(content=combined[:4000])
                result = llm.invoke(prompt)
                return sub_matches, result.content.strip()
    return [], "❌ 요청 형식을 이해하지 못했습니다. 예: '4.1장 요약 해줘' 혹은 '전체'"

# ====== 퀴즈와 채점, 질의응답 등 (기존 로직 그대로 재사용) ======
def generate_quiz(content, num_questions=5, difficulty="MEDIUM"):
    """
    교재 내용을 바탕으로 num_questions 수만큼 4지선다 객관식 문제를 생성합니다.
    난이도(difficulty): 'EASY', 'MEDIUM', 'HARD'
    """
    variation = random.choice(["핵심 개념형", "응용형", "이해형", "추론형"])
    prompt = f"""
다음 교재 내용을 바탕으로 {num_questions}개의 **4지선다 객관식 문제**를 출제하세요.
**(참고: 내용은 여러 PDF 파일에서 결합된 것일 수 있습니다.)**

- 문제 난이도: {difficulty}
- 유형: {variation}
- 각 문항은 하나의 명확한 정답을 가져야 합니다.
- 모든 문항은 반드시 **보기 a), b), c), d)** 를 포함해야 합니다.
- 보기의 순서는 랜덤하게 섞습니다.
- 정답 표시나 '(정답: ...)'은 포함하지 마세요.
- 문제 중복 금지

내용:
{content[:4000]}

출력 형식 예시:
---
[연습문제]
1. 다음 중 ...에 대한 설명으로 옳은 것은 무엇인가?
a) ...
b) ...
c) ...
d) ...
2. ...
a) ...
b) ...
c) ...
d) ...
---
"""
    result = llm.invoke(prompt)
    return result.content.strip()

def run_quiz_session(content):
    try:
        # ✅ 사용자 입력 추가
        num_questions_input = input("출제할 문제 수를 입력하세요 (기본값: 5): ").strip()
        num_questions = int(num_questions_input) if num_questions_input.isdigit() else 5

        difficulty_input = input("난이도를 선택하세요 ( EASY / MEDIUM / HRAD, 기본값: MEDIUM ): ").strip()
        difficulty = difficulty_input if difficulty_input in ["EASY", "MEDIUM", "HRAD"] else "MEDIUM"

        print(f"\n🧩 {difficulty} 난이도의 문제 {num_questions}문항을 생성합니다...\n")
        quiz_text = generate_quiz(content, num_questions, difficulty)

        print("\n--- 연습문제 ---")
        quiz_body = quiz_text.split('[연습문제]')[-1].strip()
        # 정규식 수정 없음: 기존에 잘 작동함
        questions_raw = re.findall(r"\d+\.\s*(.*?)(?=\n\d+\.|\Z)", quiz_body, re.DOTALL)
        print(quiz_body)

        if not questions_raw:
            print("❌ 문제를 파싱할 수 없습니다. 생성된 문제 형식을 확인해주세요.")
            return
            
        # =========================================================
        # ✅ 점수 계산 로직 개선: 문제 수에 따라 동적으로 점수 배분
        # =========================================================
        num_total_questions = len(questions_raw)
        if num_total_questions > 0:
            # 100점 만점을 기준으로 문항당 점수 계산 (정수형)
            score_per_question = 100 // num_total_questions 
            # 나머지 점수는 마지막 문제에 합산하여 100점을 만듭니다. (여기서는 간단하게 처리)
            # 다만, 최종 출력 시 총점만 맞추면 되므로, 배점만 알려줍니다.
        else:
            score_per_question = 0
            
        print(f"\n🧩 퀴즈를 시작합니다! (각 문항은 약 {score_per_question}점)\n")

        results = []
        total_score = 0
        correct_count = 0
        # =========================================================

        for idx, q_text in enumerate(questions_raw, 1):
            q_clean = re.sub(r'^\s+|\s+$', '', q_text)
            print(f"\n문항 {idx}: {q_clean}")
            user_answer = input("당신의 답: ").strip()

            grade_prompt_text = grading_prompt.format(
                question=q_clean,
                user_answer=user_answer,
                context=content[:4000]
            )

            try:
                result = llm.invoke(grade_prompt_text)
                result_text = result.content.strip()

                is_correct_str = (re.search(r"정답여부:\s*([^\n]+)", result_text).group(1).strip()
                                    if re.search(r"정답여부:\s*([^\n]+)", result_text) else "오답")
                is_correct = "정답" in is_correct_str

                correct_answer_match = re.search(r"정답:\s*([^\n]+)", result_text, re.DOTALL)
                correct_answer = correct_answer_match.group(1).strip() if correct_answer_match else "채점 오류: 정답을 찾을 수 없음"

                # [수정] LLM의 '정답/오답' 판단을 기반으로 동적 점수 부여
                if is_correct:
                    score = score_per_question
                    correct_count += 1
                else:
                    score = 0
                
                total_score += score
                
                results.append({
                    "question": q_clean,
                    "user_answer": user_answer,
                    "correct_answer": correct_answer,
                    "is_correct": is_correct,
                    "score": score
                })

            except Exception as e:
                # ... (오류 처리 로직) ...
                print(f"❌ 문항 {idx} 채점 중 오류 발생: {e}")
                results.append({
                    "question": q_clean,
                    "user_answer": user_answer,
                    "correct_answer": "시스템 오류로 채점 불가",
                    "is_correct": False,
                    "score": 0
                })
                continue

        # =========================================================
        # ✅ 최종 점수 계산: 소수점 이하 점수를 마지막 문제에 합산
        # =========================================================
        # 100 - (문항당 점수 * 문제 수) = 미달된 점수
        # 예: 20문제 (5점씩) -> 100 - (5 * 20) = 0
        # 예: 3문제 (33점씩) -> 100 - (33 * 3) = 1점 (미달)
        
        # 여기서 최종적으로 총점을 100점에 맞춥니다.
        final_total_score = total_score
        
        # 마지막 문제에 나머지 점수를 더해 총점을 100점으로 조정
        remaining_score = 100 - (score_per_question * num_total_questions)
        
        # 마지막 정답 문제에 남은 점수를 더합니다.
        if remaining_score > 0 and correct_count > 0:
             # 마지막 정답 문항을 찾아서 점수 조정
            for res in reversed(results):
                if res['is_correct']:
                    res['score'] += remaining_score
                    final_total_score += remaining_score
                    break
        
        # 만약 맞춘 문제가 하나도 없더라도 0점으로 정확히 표시
        if correct_count == 0:
            final_total_score = 0
        
        # 최종 결과 출력
        print("\n\n--- 연습문제 결과 ---")
        for idx, res in enumerate(results, 1):
            # ... (개별 결과 출력 로직) ...
            print("-" * 40)
            print(f"문항 {idx}: {res['question']}")
            print(f"당신의 답: {res['user_answer']}")
            print(f"정답: {res['correct_answer']}")
            result_symbol = "✅ (정답)" if res['is_correct'] else "❌ (틀림)"
            print(f"결과: {result_symbol}")
            print(f"점수: {res['score']}점") # 최종 점수를 정확히 표시
            
        print("-" * 40)
        print(f"\n=== 🏁 최종 결과 ===")
        print(f"총 {correct_count}문제 정답 / {num_total_questions}문제 ({final_total_score}점)")
        print("-" * 40)

    except Exception as e:
        print(f"❌ 연습문제 세션 오류: {e}")


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

# ====== 메인 ======
def main():
    print("=== PDF 학습 도우미 (다중 파일 지원) ===")
    pdf_paths_input = input("PDF 파일 경로들을 쉼표(,)로 구분하여 입력하세요 (예: file1.pdf, file2.pdf): ").strip()
    if not pdf_paths_input:
        print("❌ 입력된 파일 경로가 없습니다.")
        return
    pdf_paths = [p.strip() for p in pdf_paths_input.split(',') if p.strip()]
    if not pdf_paths:
        print("❌ 유효한 파일 경로가 입력되지 않았습니다.")
        return

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
        print("4. 단원별 요약 (전체 단원 요약 또는 특정 단원 요청 가능)")
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
                # 사용자에게 '전체' 또는 특정 단원(예: '4.1장 요약 해줘', '4장') 입력 받음
                req = input("요청을 입력하세요 (예: '전체' 또는 '4.1장 요약해줘', '4장'): ").strip()
                if not req:
                    print("❌ 입력이 없습니다. '전체' 또는 '4.1장' 등의 형식으로 입력하세요.")
                    continue

                if req.lower() in ["전체", "all", "모두"]:
                    print("\n--- 단원별 전체 요약 ---")
                    chapters, summaries = summarize_by_subchapter(content)
                    if not summaries:
                        print("❌ 단원 분리가 불가능하거나 단원이 없습니다.")
                    else:
                        for ch, summ in summaries.items():
                            print(f"\n[{ch} 단원 요약]\n{summ}\n")
                else:
                    found_keys, summ = summarize_requested_subchapter(content, req)
                    if not found_keys:
                        print(summ)  # 오류 메시지
                    else:
                        print(f"\n--- 요약 ({', '.join(found_keys)}) ---\n")
                        print(summ)
            except Exception as e:
                print(f"단원별 요약 중 오류: {e}")

        elif choice == "0":
            print("프로그램을 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다. 다시 입력하세요.")

if __name__ == "__main__":
    main()
