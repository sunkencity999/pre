# Makefile for PRE — Personal Reasoning Engine
#
# Targets:
#   make              — build PRE CLI binary
#   make pre          — build PRE CLI only
#   make telegram     — build Telegram bot bridge
#   make clean        — remove build artifacts
#   make install      — symlink pre-launch into ~/.local/bin

CC = clang
CFLAGS = -O3 -march=native -flto -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK
LDFLAGS = -flto -lpthread -lcompression

PRE_TARGET = pre
PRE_SRC = pre.m

TG_TARGET = pre-telegram
TG_SRC = telegram.m

.PHONY: all clean install telegram

all: $(PRE_TARGET)

$(PRE_TARGET): $(PRE_SRC) linenoise.c linenoise.h
	$(CC) $(CFLAGS) -framework Foundation -framework AppKit -framework WebKit $(PRE_SRC) linenoise.c -o $(PRE_TARGET)

$(TG_TARGET): $(TG_SRC)
	$(CC) $(CFLAGS) -framework Foundation $(TG_SRC) -o $(TG_TARGET)

telegram: $(TG_TARGET)

clean:
	rm -f $(PRE_TARGET) $(TG_TARGET)

install: $(PRE_TARGET)
	@mkdir -p $(HOME)/.local/bin
	@chmod +x pre-launch
	ln -sf $(CURDIR)/pre-launch $(HOME)/.local/bin/pre-launch
	@echo ""
	@echo "Installed: pre-launch -> $(CURDIR)/pre-launch"
	@echo "Make sure ~/.local/bin is in your PATH:"
	@echo '  export PATH="$$HOME/.local/bin:$$PATH"'
