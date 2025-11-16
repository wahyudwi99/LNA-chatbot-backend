FROM python:3.9-slim


WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY download_embedding_model.py .
RUN download_embedding_model.py

COPY api.py .
COPY module.py .
COPY template_prompt.py .
COPY queries /app/queries

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]