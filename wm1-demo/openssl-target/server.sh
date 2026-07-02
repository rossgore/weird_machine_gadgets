#!/bin/bash
set -e

CERT=${1:-cert.pem}
KEY=${2:-key.pem}
PORT=${3:-4433}

if [ ! -f "$CERT" ]; then
  openssl req -x509 -newkey rsa:2048 -keyout "$KEY" -out "$CERT" \
    -days 3 -nodes -subj "/CN=wm1-demo.local"
fi

echo "[*] Starting openssl s_server on port $PORT (TLS1.0-1.2, renegotiation enabled)"
echo "[*] Interactive mode: type 'R' + Enter at any time to send a HelloRequest"
openssl s_server -accept "$PORT" -cert "$CERT" -key "$KEY" \
  -no_tls1_3
