FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[all]"

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY examples/ ./examples/

# Install the package
RUN pip install --no-cache-dir -e .

# Expose ports
EXPOSE 8000 8501

# Default command
CMD ["knowprobe", "--help"]

# --- Production stage ---
FROM base AS production

ENV KNOWPROBE_ENV=production
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "knowprobe.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Development stage ---
FROM base AS dev

ENV KNOWPROBE_ENV=development

RUN pip install --no-cache-dir -e ".[dev]"

CMD ["bash"]
