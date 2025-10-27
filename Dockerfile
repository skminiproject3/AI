# ======================================
# 1️⃣ 빌드 단계 (Dependencies Install)
# ======================================
FROM python:3.12-slim AS builder
WORKDIR /app

# 기본 툴 설치 (빌드용 패키지 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

# ======================================
# 2️⃣ 실행 단계 (Runtime Environment)
# ======================================
FROM python:3.12-slim
WORKDIR /app

# 경량 필수 모듈만 별도 설치 (pymysql, dotenv 등)
RUN pip install --no-cache-dir python-multipart==0.0.20 pymysql==1.1.1 python-dotenv==1.0.1

# 빌드된 라이브러리 복사
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# 코드 복사
COPY . .

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/uploaded_pdfs /app/vector_stores \
    && chown -R appuser:appuser /app/uploaded_pdfs /app/vector_stores

# ✅ 이후부터 appuser로 실행
USER appuser

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# ======================================
# 3️⃣ 실행 명령
# ======================================
CMD ["uvicorn", "main_vector:app", "--host", "0.0.0.0", "--port", "8000"]
