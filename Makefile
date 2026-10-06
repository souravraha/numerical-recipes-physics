# Top-level orchestrator for the two-language repo.
#
# Each language owns its own build (Python via pixi/pytest, C via c/Makefile);
# this file gives one entry point so the whole project builds and tests the
# same way:
#
#   make            ->  run every C and Python test
#   make c-test     ->  C port only         (c/)
#   make py-test    ->  Python package only (src/ + tests/)
#   make clean      ->  remove C build artifacts
#
# The Python suite needs numpy, scipy, and pytest. If a pixi environment
# exists it is used automatically; otherwise `pytest` is expected on PATH
# (e.g. after `pixi shell`, or a plain `pip install numpy scipy pytest`).

# Prefer the committed pixi env's pytest if present, else a `pytest` on PATH.
PYTEST ?= $(if $(wildcard .pixi/envs/*/bin/pytest),$(firstword $(sort $(wildcard .pixi/envs/*/bin/pytest))),pytest)

.DEFAULT_GOAL := help

.PHONY: help test all c c-build c-test py py-test clean
.PHONY: help test all c c-build c-test py py-test examples clean

help:
	@echo "numerical-recipes-physics -- two-language build/test entry point"
	@echo ""
	@echo "  make test      run the C and Python test suites (default)"
	@echo "  make c-test    build + test the C port (c/)"
	@echo "  make py-test   run the Python test suite (needs numpy, scipy, pytest)"
	@echo "  make examples  run the atlas case studies (figures -> output/)"
	@echo "  make clean     remove C build artifacts (c/build/)"

all: test

test: c-test py-test

c-build:
	$(MAKE) -C c

c-test:
	$(MAKE) -C c test

PYBIN = $(if $(wildcard .pixi/envs/*/bin/python),$(firstword $(sort $(wildcard .pixi/envs/*/bin/python))),$(shell command -v python3))

py-test:
	if [ -n "$(wildcard .pixi/envs/*/bin/pytest)" ]; then \
	  PYTHONPATH=src $(firstword $(sort $(wildcard .pixi/envs/*/bin/pytest))) tests/ -q; \
	else \
	  PYTHONPATH=src $(PYTEST) tests/ -q; \
	fi

examples:
	PYTHONPATH=src ./examples/run_examples.sh

clean:
	$(MAKE) -C c clean
	@rm -rf .pytest_cache
