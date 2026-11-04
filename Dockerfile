# syntax=docker/dockerfile:1
# --- build stage: install wheels into a venv ---------------------------------------------
FROM python:3.11-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

# --- runtime stage: slim, non-root, no secrets baked in ------------------------------------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/opt/venv/bin:$PATH" HOME=/tmp DATA_DIR=/app/data
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /tmp --shell /usr/sbin/nologin app \
 && mkdir -p /app/data && chown app:app /app/data
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY app ./app
COPY eval/questions.json eval/__init__.py ./eval/
USER app
EXPOSE 8000
VOLUME ["/app/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "app.main:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
