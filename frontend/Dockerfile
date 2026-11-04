# syntax=docker/dockerfile:1
# Streamlit UI image (build context = project root:  docker build -f frontend/Dockerfile .)
FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
RUN groupadd --system --gid 10001 ui \
 && useradd --system --uid 10001 --gid ui --home-dir /tmp --shell /usr/sbin/nologin ui
WORKDIR /app
COPY frontend/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY frontend ./frontend
COPY .streamlit ./.streamlit
COPY docs/architecture.png ./docs/architecture.png
USER ui
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).status == 200 else 1)"
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
