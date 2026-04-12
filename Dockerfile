FROM ghcr.io/astral-sh/uv:python3.11-bookworm

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml ./
COPY server.py ./
COPY agent agent
COPY core core

RUN --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync

ENTRYPOINT ["uv", "run", "server.py"]
CMD ["--host", "0.0.0.0", "--port", "9018"]
EXPOSE 9018
