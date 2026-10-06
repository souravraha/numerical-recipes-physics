CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Iinclude
LDFLAGS ?= -lm

SRC := $(wildcard src/*.c)
OBJ := $(SRC:src/%.c=build/%.o)
LIB := build/libnr.a
TESTS := $(patsubst tests/%.c,build/tests/%,$(wildcard tests/test_*.c))

all: $(LIB)

$(LIB): $(OBJ)
	@mkdir -p $(dir $@)
	ar rcs $@ $(OBJ)

build/%.o: src/%.c $(wildcard include/nr/*.h)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

build/tests/%: tests/%.c $(LIB) tests/test_harness.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

test: $(TESTS)
	@for t in $(TESTS); do echo "== $$t"; "$$t" || exit 1; done
	@echo "ALL TESTS PASSED"

clean:
	rm -rf build

.PHONY: all test clean
