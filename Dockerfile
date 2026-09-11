FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY config /app/config
COPY datasets evaluation filters learned models runners state_estimation utils /app/

RUN python -m pip install --no-cache-dir .

COPY examples /app/examples

CMD ["python", "examples/basic_filter_loop.py"]
