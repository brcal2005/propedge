FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
# Bets, odds and game logs live here; mount a persistent disk at /data
ENV PROPEDGE_HOME=/data
EXPOSE 8000
CMD ["propedge", "web", "--host", "0.0.0.0", "--no-browser"]
