# WM-1: OpenSSL Transcript Sentinel

A working demonstration of a beneficial weird machine built from
standard, unmodified OpenSSL public API calls. WM-1 turns ordinary TLS
handshake bookkeeping into an integrity monitor that detects
mid-connection parameter tampering, including renegotiation.

No OpenSSL source is patched. No instrumentation is injected. Every gadget
used here is a public OpenSSL function used to produce a security feature the API was
not designed to provide.

---

## 1. Background: What Is a Weird Machine, and Why Is This One "Beneficial"?

A weird machine is a system that performs computation its designers never
intended, by chaining together legitimate operations in an order or context
the original implementation never anticipated. The term is most often used
in offensive security. Attackers build weird machines out of memory
corruption primitives to achieve arbitrary code execution using only
"legal" CPU instructions.

Here we apply the same idea defensively. It does not exploit a memory safety
bug or protocol flaw. Instead, it exploits an architectural side effect
of how OpenSSL manages handshake state internally: OpenSSL leaves
handshake results (cipher, protocol version, peer certificate) readable on
the live `SSL` object indefinitely after the handshake completes, and it
exposes a callback (`info_callback`) that fires at every state transition,
including renegotiation.

## 2. Why This Works in OpenSSL

WM-1 depends on three pieces of state being simultaneously true: (1) the handshake has been marked complete, (2) the application
has not yet consumed any post-handshake data, and (3) the negotiated
parameters are still readable on the connection object. OpenSSL satisfies
all three because of how it manages the `SSL` struct's internal fields.

OpenSSL treats handshake completion as a transition event. The state
persists, and the system moves on. 

## 3. TLS and OpenSSL Version Scope

| Dimension | In scope | Out of scope |
|---|---|---|
| TLS version | TLS 1.0, TLS 1.1, TLS 1.2 (all support mid-connection renegotiation) | TLS 1.3 (renegotiation removed from the protocol entirely, replaced by post-handshake auth) |
| OpenSSL version | 1.0.2 through 1.1.1 (primary target); technically present through 3.x since legacy TLS and renegotiation remain in the codebase, deprecated but not removed | N/A |

This demo pins OpenSSL 1.1.1w and exercises TLS 1.2.

## 4. The Gadget Chain

WM-1 is assembled from six OpenSSL primitives, none of which were designed
to work together. It is chained so that each one's output becomes the next
one's input:

1. **`info_callback` registered via `SSL_CTX_set_info_callback()`** — a
   transition-frequency hook. Fires on every handshake state change,
   including `SSL_CB_HANDSHAKE_START` and `SSL_CB_HANDSHAKE_DONE`.
2. **Post-handshake state persistence on `s->s3`** — the OpenSSL-specific
   architectural property that negotiated cipher, version, and peer cert
   remain readable long after the handshake that produced them completes.
3. **`SSL_set_ex_data()` / `SSL_get_ex_data()`** — OpenSSL's external data
   slot mechanism, originally intended for application bookkeeping, here
   repurposed to persist WM-1's fingerprint snapshot across the life of
   the connection without modifying the `SSL` struct itself.
4. **`wm1_SSL_read()` wrapper** — intercepts every application read,
   re-derives the live fingerprint, and compares it against the armed
   snapshot before releasing data to the caller.
5. **Synthetic `SSL_ERROR_WANT_READ` surfacing** — when a renegotiation is
   detected mid-stream, the read wrapper stalls the caller by returning a
   retryable error instead of application data, while internally pumping
   `SSL_do_handshake()` to drive the renegotiation to completion.
6. **Second `SSL_CB_HANDSHAKE_START` as a renegotiation trigger** — the
   same callback registered in gadget 1 fires again mid-connection when a
   peer-initiated `HelloRequest` arrives, which is the signal WM-1 uses to
   transition into its stall-and-reverify state.

```
info_callback fires SSL_CB_HANDSHAKE_DONE
        |
   s->s3 still populated (OpenSSL only)
        |
SSL_get_current_cipher() / SSL_version() / SSL_get_peer_certificate()
        |
SHA-256 SPKI pin + cipher id + version -> fingerprint
        |
SSL_set_ex_data() stores fingerprint on the SSL object
        |
wm1_SSL_read() wrapper armed
        |
every SSL_read() re-derives fingerprint, compares to snapshot
        |
if SSL_CB_HANDSHAKE_START fires again -> WM1_STATE_RENEG
        |
wm1_SSL_read() stalls, pumps SSL_do_handshake()
        |
SSL_CB_HANDSHAKE_DONE fires again -> re-capture, compare to original
        |
match -> silent continue    mismatch -> wm1_alert()
```

## 5. Three Phases of Operation

**Phase 1 — Fingerprint Capture.** At the moment `SSL_CB_HANDSHAKE_DONE`
fires, WM-1 snapshots the cipher suite, TLS version, and a SHA-256 pin of
the peer certificate's SubjectPublicKeyInfo, storing them via
`SSL_set_ex_data()`.

**Phase 2 — Read-Path Monitoring.** Every subsequent `SSL_read()` call is
wrapped so that live parameters are re-derived and compared against the
Phase 1 snapshot before data is returned to the application. Any
divergence triggers `wm1_alert()`.

**Phase 3 — Renegotiation Detection.** A second `SSL_CB_HANDSHAKE_START`
after the sentinel is armed signals an in-progress renegotiation. The read
wrapper stalls app data delivery, pumps the handshake to completion, and
re-verifies the resulting fingerprint against the original snapshot before
resuming normal read-path monitoring.

---

## 6. Architecture of This Demo

```
wm1-demo/
├── README.md                      (this file)
├── docker-compose.yml
└── openssl-target/
    ├── Dockerfile                  # pinned OpenSSL 1.1.1w build
    ├── server.sh                   # interactive s_server target
    └── wm1_sentinel/
        ├── main.c                  # the sentinel itself (the weird machine)
        └── Makefile
```

The target server runs real, pinned OpenSSL 1.1.1w inside a container. The
sentinel is a standalone C client, compiled against whatever OpenSSL
headers are available on the host, that connects to the target and
performs all three phases described above. Every log line the sentinel prints corresponds to a real OpenSSL API call
returning real, live connection state.

---

## 7. Running the Demo

### Prerequisites

- Docker (for the pinned OpenSSL 1.1.1w target server)
- A local OpenSSL development install (headers + libs) to compile the
  sentinel — `libssl-dev` on Debian/Ubuntu, `openssl-devel` on
  RHEL/Fedora, or Homebrew's `openssl@3` on macOS
- `pkg-config` able to locate OpenSSL, or a manually specified `-I`/`-L`
  path in the Makefile

### Step 1 — Build and Start the Target Server

```sh
cd wm1-demo
docker compose build
docker compose run --rm -p 4433:4433 openssl-target ./server.sh
```

Using `docker compose run` instead of `up` is required — it attaches your
terminal's stdin directly to the container process, which is necessary in
Step 4 to send renegotiation triggers interactively.

Expected output:

```
Generating a RSA private key
.....+++++
writing new private key to 'key.pem'
-----
[*] Starting openssl s_server on port 4433 (TLS1.0-1.2, renegotiation enabled)
[*] Interactive mode: type 'R' + Enter at any time to send a HelloRequest
Using default temp DH parameters
ACCEPT
```

The self-signed cert/key pair is regenerated on every fresh container run
since the container filesystem is ephemeral — this is expected and not an
error.

### Step 2 — Build the Sentinel

In a second terminal:

```sh
cd wm1-demo/openssl-target/wm1_sentinel
make
```

### Step 3 — Connect and Confirm Phases 1 and 2

```sh
./wm1_sentinel 127.0.0.1 4433 --verbose
```

Expected output:

```
[WM1] fingerprint armed (cipher=ECDHE-RSA-AES256-GCM-SHA384 version=0x303)
[WM1] connected. cipher=ECDHE-RSA-AES256-GCM-SHA384 version=0x303
[WM1] read-path check OK (4095 bytes)
...
[WM1] session ended. violations=0
```

The "fingerprint armed" line confirms Phase 1 fired correctly at handshake
completion. Each "read-path check OK" line confirms Phase 2 is
re-verifying live state on every single `SSL_read()` call, not just once.
`violations=0` is the expected clean-session result. Cross-check the
cipher and version shown here against the server's own diagnostic output
(`New, TLSv1.2, Cipher is ...`) — they should match exactly, confirming
the sentinel's fingerprint is grounded in the real negotiated session, not
fabricated.

### Step 4 — Trigger Renegotiation and Confirm Phase 3

Reconnect the sentinel (it exits after the first `-www`-less session
closes, or leave it running if the server does not close the connection):

```sh
./wm1_sentinel 127.0.0.1 4433 --verbose
```

Switch to Terminal 1 (the server) and type:

```
R
```

then press Enter. Repeat as many times as desired.

Expected server output per trigger:

```
R
SSL_do_handshake -> 1
Read BLOCK
```

Expected sentinel output per trigger:

```
[WM1] renegotiation START detected -> WM1_STATE_RENEG
[WM1] renegotiation re-verified OK, parameters unchanged
```

Each `R` on the server should produce exactly one matching pair of lines
on the sentinel side. This confirms the full Phase 3 chain: the second
`SSL_CB_HANDSHAKE_START` fired, `wm1_SSL_read()` correctly stalled instead
of returning data, `SSL_do_handshake()` drove the renegotiation to
completion internally, and the re-armed fingerprint matched the original
snapshot (expected, since the server did not change its cert or cipher).

Stop the sentinel with `Ctrl-C` when done; the server will log a benign
`ERROR` / `shutting down SSL` / `CONNECTION CLOSED` sequence — this is the
expected result of an abrupt client disconnect, not a protocol failure.

### Step 5 — Independent Verification via pcap

To confirm the renegotiation occurred on the wire, independent of the
sentinel's own reporting, start a capture **before** Step 1:

```sh
sudo tcpdump -i lo0 -w wm1_reneg.pcap port 4433
```

(Use `lo` instead of `lo0` on Linux.) Run the full sequence above, then
stop the capture and open `wm1_reneg.pcap` in Wireshark with a `tls`
display filter. You should see the original ClientHello/ServerHello/
Certificate/Finished sequence, followed by a `Hello Request` message, and
a second ClientHello/ServerHello/Finished sequence layered inside the
already-established record stream. That is the on-wire signature of in-band
renegotiation. 

### Step 6  Force a Mismatch to Confirm Alerting

Everything above confirms the sentinel correctly does *nothing* when
nothing is wrong. To confirm it correctly detects an actual problem,
restart the target server with a different certificate (delete `cert.pem`
and `key.pem` so `server.sh` regenerates a new one, or point `-cert`/`-key`
at a different pair) between two renegotiation triggers against the same
sentinel connection. Expected output:

```
[WM1-ALERT] post-renegotiation parameters differ from original snapshot
```

with the sentinel's exit code changing from `0` to `2`.

---
