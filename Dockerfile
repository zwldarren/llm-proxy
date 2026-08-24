# Stage 1: Build frontend
FROM oven/bun:1 AS frontend-builder
WORKDIR /app/frontend

# Copy dependency files and patches
COPY frontend/package.json frontend/bun.lock ./
COPY frontend/patches patches/

# Install dependencies
RUN bun install

# Copy source code
COPY frontend/ .

# Build frontend
RUN bun run build

# Stage 2: Build backend
FROM python:3.14-slim

# Copy bun tools (for bunx command) from first stage
COPY --from=frontend-builder /usr/local/bin/bun /usr/local/bin/bun
COPY --from=frontend-builder /usr/local/bin/bunx /usr/local/bin/bunx

# Copy uv tools (for uvx command)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Copy the project into the image
COPY . /app

# Copy built frontend assets
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Expose port
EXPOSE 8080

# Default command
CMD ["uv", "run", "llm-proxy"]
