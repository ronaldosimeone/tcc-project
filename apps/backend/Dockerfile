FROM python:3.11

WORKDIR /app
COPY . .

RUN pip install fastapi uvicorn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]