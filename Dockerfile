FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/data /app/backtest/results
ENV DATA_PATH=/app/data
ENV PAPER_MODE=true
ENV PORT=8080
EXPOSE $PORT
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
