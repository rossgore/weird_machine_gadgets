
# Pinned OpenSSL 1.1.1w build for the WM-1 target server.
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y build-essential wget perl
WORKDIR /src
RUN wget https://www.openssl.org/source/openssl-1.1.1w.tar.gz && \
    tar xzf openssl-1.1.1w.tar.gz
RUN cd openssl-1.1.1w && ./config && make -j$(nproc) && make install_sw && make install_ssldirs
ENV LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib
WORKDIR /app
COPY server.sh .
RUN chmod +x server.sh
EXPOSE 4433
CMD ["./server.sh"]
