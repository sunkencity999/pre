# Makefile for PRE — Personal Reasoning Engine
#
# Targets:
#   make              — build infer + pre binaries
#   make infer        — build inference server only
#   make pre          — build PRE CLI only
#   make chat         — build lightweight chat client
#   make clean        — remove build artifacts
#   make install      — symlink pre-launch into ~/.local/bin
#
# Metal shaders are compiled from source at runtime via
# MTLDevice newLibraryWithSource:, so no offline metal compiler needed.

CC = clang
CFLAGS = -O3 -march=native -flto -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate
LDFLAGS = -flto -lpthread -lcompression

INFER_TARGET = infer
INFER_SRC = infer.m

PRE_TARGET = pre
PRE_SRC = pre.m

CHAT_TARGET = chat
CHAT_SRC = chat.m

BENCH_TARGET = metal_infer
BENCH_SRC = main.m
SHADER_SRC = shaders.metal

.PHONY: all clean install build-pre build-infer build-chat

all: $(INFER_TARGET) $(PRE_TARGET)

$(INFER_TARGET): $(INFER_SRC) $(SHADER_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(INFER_SRC) -o $(INFER_TARGET)

$(PRE_TARGET): $(PRE_SRC) linenoise.c linenoise.h
	$(CC) $(CFLAGS) -framework Foundation $(PRE_SRC) linenoise.c -o $(PRE_TARGET)

$(CHAT_TARGET): $(CHAT_SRC) linenoise.c linenoise.h
	$(CC) $(CFLAGS) -framework Foundation $(CHAT_SRC) linenoise.c -o $(CHAT_TARGET)

$(BENCH_TARGET): $(BENCH_SRC) $(SHADER_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(BENCH_SRC) -o $(BENCH_TARGET)

build-infer: $(INFER_TARGET)
build-pre: $(PRE_TARGET)
build-chat: $(CHAT_TARGET)

clean:
	rm -f $(INFER_TARGET) $(PRE_TARGET) $(CHAT_TARGET) $(BENCH_TARGET)

install: $(PRE_TARGET)
	@mkdir -p $(HOME)/.local/bin
	@chmod +x pre-launch
	ln -sf $(CURDIR)/pre-launch $(HOME)/.local/bin/pre-launch
	@echo ""
	@echo "Installed: pre-launch -> $(CURDIR)/pre-launch"
	@echo "Make sure ~/.local/bin is in your PATH:"
	@echo '  export PATH="$$HOME/.local/bin:$$PATH"'
