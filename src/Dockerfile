FROM python:3.11-slim

WORKDIR /app

# Node.js is needed for npx (firecrawl-mcp stdio transport)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

RUN mkdir -p /app/cache
ENV CACHE_DB_PATH=/app/cache/cache.db

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["python", "src/main.py"]
