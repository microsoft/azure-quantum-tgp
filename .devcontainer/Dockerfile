FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

COPY . /tmp/package/
RUN /opt/conda/bin/pip install /tmp/package/ && rm -rf /tmp/package
