# backend/Dockerfile
# Build stage
FROM python AS builder

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --user -r requirements.txt

# Runtime stage
FROM python

WORKDIR /app
ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH

# Copy from builder
COPY --from=builder /root/.local /root/.local
COPY ./app /app/app
# COPY ./model /app/model

# Ensure scripts in .local are usable
RUN chmod -R 755 /root/.local && \
    apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]