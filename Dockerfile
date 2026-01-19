FROM python:3.13-slim

ENV PYTHONUNBUFFERED=TRUE

RUN pip --no-cache-dir install uv

WORKDIR /app

COPY . .

RUN uv sync

EXPOSE 8000

ENTRYPOINT ["uv", "run", "fastapi", "run", "predict.py"]
