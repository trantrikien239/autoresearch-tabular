FROM node:20-slim

WORKDIR /app

# System deps: Python, git, curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        git curl libgomp1 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Git config
RUN git config --global user.email "agent@autoresearch" && \
    git config --global user.name "autoresearch-agent"

# Create non-root user (Claude Code refuses --dangerously-skip-permissions as root)
RUN useradd -m -u 1001 -s /bin/bash researcher
RUN chown -R researcher:researcher /app

USER researcher
WORKDIR /app/code

CMD ["bash"]
