FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
WORKDIR /app

RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

COPY requirements-fixed.txt .
RUN pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements-fixed.txt

COPY . .
RUN make install && make test

EXPOSE 8888
CMD ["make", "train"]
