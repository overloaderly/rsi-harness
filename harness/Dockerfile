# Harness 镜像：python 3.12 + 官方 contest_harness SDK 1.0.2 + 选手模块
#
# 构建（必须是 linux/amd64）：
#   docker build --platform linux/amd64 -t rsi-harness:v1 .
# 导出（平台只收 docker save 的未压缩单镜像 tar）：
#   docker save rsi-harness:v1 -o rsi-harness-v1.tar
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# pip 源：默认官方 PyPI；国内构建可传 --build-arg PIP_INDEX_URL=...
ARG PIP_INDEX_URL=https://pypi.org/simple

# 官方 SDK（来自 contest-harness-sdk-1.0.2.tar.gz），装进系统 site-packages
WORKDIR /opt/contest-harness
COPY vendor/README.md vendor/pyproject.toml /opt/contest-harness/
COPY vendor/contest_harness /opt/contest-harness/contest_harness
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" --timeout 60 --retries 5 /opt/contest-harness numpy \
    && rm -rf /opt/contest-harness

# 选手侧文件：入口模板 + 提交模块 + 参考模型实现
WORKDIR /app
COPY harness.py /app/harness.py
COPY contest_submission.py rsi_model.py rsi_data.py /app/

# 离线自测脚本（不在 /app，避免被平台当作提交模块）
COPY selftest.py /opt/harness-selftest/selftest.py

USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]
