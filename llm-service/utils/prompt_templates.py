# 예상문제 생성용 프롬프트
QUIZ_GENERATION_PROMPT = """
다음 학습자료를 바탕으로 5개의 객관식 문제를 생성하세요.
형식: JSON (id, question, options, answer)
학습자료:
{content}
"""

# 사용자 답안 채점용 프롬프트
QUIZ_GRADING_PROMPT = """
사용자가 푼 답안을 기반으로 채점하세요.
정답: {answer}
사용자 답안: {user_answer}
점수: 0~100 점
피드백: 2~3 문장
JSON 형식으로 반환
"""
