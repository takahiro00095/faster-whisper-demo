# 最新の faster-whisper が要求する CUDA 12 + cuDNN 9 のイメージを使用
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 環境変数を設定
ENV DEBIAN_FRONTEND=noninteractive

# NVIDIAのイメージによっては標準パスに含まれていないことがあるため明示的にパスを通す
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# blinkerはシステムパッケージとの衝突を回避するために ignore-installed を使うので、requirements.txtの外側でインストール
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]