/*
 * WM-1: OpenSSL Transcript Sentinel
 * ----------------------------------
 * Standalone demo/prototype of the WM-1 beneficial weird machine.
 * Built strictly against real OpenSSL public API (no forks, no patches).
 *
 * Pinned target: OpenSSL 1.1.1w
 * TLS versions exercised: TLS 1.0 - TLS 1.2 (renegotiation-capable)
 *
 * Phases:
 *   1. Fingerprint capture at SSL_CB_HANDSHAKE_DONE
 *   2. Read-path monitoring (every SSL_read re-verifies live state)
 *   3. Renegotiation detection + stall + re-verify
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/x509.h>
#include <openssl/evp.h>

#define WM1_EX_DATA_IDX_NAME "wm1_sentinel_idx"

typedef enum {
    WM1_STATE_UNARMED = 0,
    WM1_STATE_ARMED,
    WM1_STATE_RENEG
} wm1_state_t;

typedef struct {
    int cipher_id;
    char cipher_name[128];
    int version;
    unsigned char spki_sha256[32];
    int has_cert;
    long verify_result;
} wm1_fingerprint_t;

typedef struct {
    wm1_state_t state;
    wm1_fingerprint_t snapshot;
    int strict;
    int verbose;
    int violations;
} wm1_sentinel_t;

static int g_wm1_ex_idx = -1;

/* ---- Gadget: SPKI pinning via live peer cert read ---- */
static int wm1_compute_spki_pin(SSL *ssl, unsigned char out[32]) {
    X509 *cert = SSL_get_peer_certificate(ssl); /* live read of ssl->session->peer */
    if (!cert) return 0;
    unsigned char *der = NULL;
    int len = i2d_X509_PUBKEY(X509_get_X509_PUBKEY(cert), &der);
    if (len <= 0) { X509_free(cert); return 0; }
    unsigned int md_len = 0;
    EVP_MD_CTX *mctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(mctx, EVP_sha256(), NULL);
    EVP_DigestUpdate(mctx, der, (size_t)len);
    EVP_DigestFinal_ex(mctx, out, &md_len);
    EVP_MD_CTX_free(mctx);
    OPENSSL_free(der);
    X509_free(cert);
    return 1;
}

/* ---- Gadget: fingerprint capture from live post-handshake state ---- */
static void wm1_capture_fingerprint(SSL *ssl, wm1_fingerprint_t *fp) {
    memset(fp, 0, sizeof(*fp));
    const SSL_CIPHER *c = SSL_get_current_cipher(ssl); /* reads ssl->s3->tmp.new_cipher path */
    if (c) {
        fp->cipher_id = (int)SSL_CIPHER_get_id(c);
        strncpy(fp->cipher_name, SSL_CIPHER_get_name(c), sizeof(fp->cipher_name) - 1);
    }
    fp->version = SSL_version(ssl); /* reads ssl->version */
    fp->has_cert = wm1_compute_spki_pin(ssl, fp->spki_sha256);
    fp->verify_result = SSL_get_verify_result(ssl);
}

static int wm1_fingerprint_matches(const wm1_fingerprint_t *a, const wm1_fingerprint_t *b) {
    if (a->cipher_id != b->cipher_id) return 0;
    if (a->version != b->version) return 0;
    if (a->has_cert != b->has_cert) return 0;
    if (a->has_cert && memcmp(a->spki_sha256, b->spki_sha256, 32) != 0) return 0;
    return 1;
}

static void wm1_alert(wm1_sentinel_t *s, const char *reason) {
    s->violations++;
    fprintf(stderr, "[WM1-ALERT] %s\n", reason);
}

/* ---- Gadget: info_callback — the transition-frequency hook ---- */
static void wm1_info_callback(const SSL *ssl_c, int where, int ret) {
    (void)ret;
    SSL *ssl = (SSL *)ssl_c;
    wm1_sentinel_t *s = (wm1_sentinel_t *)SSL_get_ex_data(ssl, g_wm1_ex_idx);
    if (!s) return;

    if (where & SSL_CB_HANDSHAKE_START) {
        if (s->state == WM1_STATE_ARMED) {
            if (s->verbose) fprintf(stderr, "[WM1] renegotiation START detected -> WM1_STATE_RENEG\n");
            s->state = WM1_STATE_RENEG;
        }
    }

    if (where & SSL_CB_HANDSHAKE_DONE) {
        wm1_fingerprint_t fresh;
        wm1_capture_fingerprint(ssl, &fresh); /* still valid: s->s3 populated */

        if (s->state == WM1_STATE_UNARMED) {
            s->snapshot = fresh;
            s->state = WM1_STATE_ARMED;
            if (s->verbose) fprintf(stderr, "[WM1] fingerprint armed (cipher=%s version=0x%x)\n",
                                     fresh.cipher_name, fresh.version);
        } else if (s->state == WM1_STATE_RENEG) {
            if (!wm1_fingerprint_matches(&s->snapshot, &fresh)) {
                wm1_alert(s, "post-renegotiation parameters differ from original snapshot");
            } else if (s->verbose) {
                fprintf(stderr, "[WM1] renegotiation re-verified OK, parameters unchanged\n");
            }
            s->snapshot = fresh;
            s->state = WM1_STATE_ARMED;
        }
    }
}

/* ---- Gadget: wm1_SSL_read wrapper — read-path monitoring + stall ---- */
static int wm1_SSL_read(SSL *ssl, wm1_sentinel_t *s, void *buf, int num) {
    if (s->state == WM1_STATE_RENEG) {
        int hs = SSL_do_handshake(ssl);
        if (hs != 1) {
            int err = SSL_get_error(ssl, hs);
            if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
                errno = EAGAIN;
                return -1;
            }
        }
    }

    int n = SSL_read(ssl, buf, num);
    if (n > 0) {
        wm1_fingerprint_t live;
        wm1_capture_fingerprint(ssl, &live);
        if (!wm1_fingerprint_matches(&s->snapshot, &live)) {
            wm1_alert(s, "live read-path parameters diverge from armed snapshot");
            if (s->strict) {
                SSL_shutdown(ssl);
                return -1;
            }
        } else if (s->verbose) {
            fprintf(stderr, "[WM1] read-path check OK (%d bytes)\n", n);
        }
    }
    return n;
}

static SSL_CTX *wm1_make_ctx(void) {
    SSL_CTX *ctx = SSL_CTX_new(TLS_client_method());
    SSL_CTX_set_min_proto_version(ctx, TLS1_VERSION);
    SSL_CTX_set_max_proto_version(ctx, TLS1_2_VERSION);
    SSL_CTX_set_info_callback(ctx, wm1_info_callback);
    return ctx;
}

static int wm1_connect_tcp(const char *host, const char *port) {
    struct addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) != 0) return -1;
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0 || connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        freeaddrinfo(res);
        return -1;
    }
    freeaddrinfo(res);
    return fd;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s host port [--strict] [--verbose]\n", argv[0]);
        return 1;
    }
    const char *host = argv[1];
    const char *port = argv[2];
    wm1_sentinel_t sentinel;
    memset(&sentinel, 0, sizeof(sentinel));
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--strict")) sentinel.strict = 1;
        if (!strcmp(argv[i], "--verbose")) sentinel.verbose = 1;
    }

    SSL_library_init();
    SSL_load_error_strings();

    g_wm1_ex_idx = SSL_get_ex_new_index(0, (void*)WM1_EX_DATA_IDX_NAME, NULL, NULL, NULL);

    SSL_CTX *ctx = wm1_make_ctx();
    SSL *ssl = SSL_new(ctx);
    SSL_set_ex_data(ssl, g_wm1_ex_idx, &sentinel);

    int fd = wm1_connect_tcp(host, port);
    if (fd < 0) { fprintf(stderr, "tcp connect failed\n"); return 1; }
    SSL_set_fd(ssl, fd);

    if (SSL_connect(ssl) != 1) {
        fprintf(stderr, "TLS handshake failed\n");
        ERR_print_errors_fp(stderr);
        return 1;
    }

    fprintf(stderr, "[WM1] connected. cipher=%s version=0x%x\n",
            sentinel.snapshot.cipher_name, sentinel.snapshot.version);

    char req[] = "GET / HTTP/1.0\r\n\r\n";
    SSL_write(ssl, req, (int)strlen(req));

    char buf[4096];
    int n;
    while ((n = wm1_SSL_read(ssl, &sentinel, buf, sizeof(buf) - 1)) > 0) {
        buf[n] = 0;
        if (sentinel.verbose) fwrite(buf, 1, n, stdout);
    }

    fprintf(stderr, "[WM1] session ended. violations=%d\n", sentinel.violations);
    SSL_shutdown(ssl);
    SSL_free(ssl);
    SSL_CTX_free(ctx);
    close(fd);

    return sentinel.violations > 0 ? 2 : 0;
}
