ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HYBRIDQA_LLM_PROVIDER=hf_local \
    HYBRIDQA_HF_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct \
    HYBRIDQA_DATA_DIR=/app/data \
    HYBRIDQA_SAMPLE_DIR=/app/data/sample \
    HYBRIDQA_INDEX_DIR=/app/data/index

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY data/sample /app/data/sample
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
