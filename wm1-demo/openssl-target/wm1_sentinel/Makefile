CC = cc
CFLAGS = -Wall -Wextra -g $(shell pkg-config --cflags openssl)
LDLIBS = $(shell pkg-config --libs openssl)

wm1_sentinel: main.c
	$(CC) $(CFLAGS) main.c -o wm1_sentinel $(LDLIBS)

clean:
	rm -f wm1_sentinel
