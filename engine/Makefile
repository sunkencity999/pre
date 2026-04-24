# Makefile for PRE — Personal Reasoning Engine
#
# Targets:
#   make              — build PRE CLI binary
#   make pre          — build PRE CLI only
#   make telegram     — build Telegram bot bridge
#   make menubar      — build PRE.app menu bar helper
#   make clean        — remove build artifacts
#   make install      — symlink pre-launch + install PRE.app

CC = clang
CFLAGS = -O3 -march=native -flto -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK
LDFLAGS = -flto -lpthread -lcompression

PRE_TARGET = pre
PRE_SRC = pre.m

TG_TARGET = pre-telegram
TG_SRC = telegram.m

MENUBAR_APP = PRE.app
MENUBAR_SRC = pre-menubar.swift
MENUBAR_BIN = $(MENUBAR_APP)/Contents/MacOS/pre-menubar

.PHONY: all clean install telegram menubar

all: $(PRE_TARGET)

$(PRE_TARGET): $(PRE_SRC) linenoise.c linenoise.h
	$(CC) $(CFLAGS) -framework Foundation -framework AppKit -framework WebKit $(PRE_SRC) linenoise.c -o $(PRE_TARGET)

$(TG_TARGET): $(TG_SRC)
	$(CC) $(CFLAGS) -framework Foundation $(TG_SRC) -o $(TG_TARGET)

telegram: $(TG_TARGET)

$(MENUBAR_BIN): $(MENUBAR_SRC)
	@mkdir -p $(MENUBAR_APP)/Contents/MacOS $(MENUBAR_APP)/Contents/Resources
	swiftc -O -o $(MENUBAR_BIN) $(MENUBAR_SRC) -framework Cocoa

menubar: $(MENUBAR_BIN)

clean:
	rm -f $(PRE_TARGET) $(TG_TARGET) $(MENUBAR_BIN)

install: $(PRE_TARGET) $(MENUBAR_BIN)
	@mkdir -p $(HOME)/.local/bin
	@chmod +x pre-launch
	ln -sf $(CURDIR)/pre-launch $(HOME)/.local/bin/pre-launch
	@mkdir -p $(HOME)/Applications
	@rm -rf $(HOME)/Applications/PRE.app
	cp -R $(CURDIR)/PRE.app $(HOME)/Applications/PRE.app
	@echo ""
	@echo "Installed: pre-launch -> $(CURDIR)/pre-launch"
	@echo "Installed: PRE.app -> ~/Applications/PRE.app"
	@echo "Make sure ~/.local/bin is in your PATH:"
	@echo '  export PATH="$$HOME/.local/bin:$$PATH"'
