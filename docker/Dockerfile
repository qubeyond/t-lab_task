FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    TOKENIZERS_PARALLELISM=false \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

ENV PATH="/app/.venv/bin:${PATH}"

COPY env/ env/
COPY memory/ memory/
COPY agents/ agents/

COPY realdata/ realdata/
COPY experiments/ experiments/
COPY analysis/ analysis/
COPY tests/ tests/
COPY report/ report/
COPY README.md .

ENTRYPOINT ["python"]
CMD ["experiments/run_tier1.py"]
