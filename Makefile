.DEFAULT_GOAL := build

.PHONY: help build local-bootstrap local-bootstrap-debug

build:
	@if [ ! -f build/Makefile ]; then \
		printf "build/ is not configured yet. Run 'make local-bootstrap' first.\n"; \
		exit 1; \
	fi
	$(MAKE) -C build

help:
	@printf "Available targets:\n"
	@printf "  build                  Build the configured build/ tree\n"
	@printf "  local-bootstrap        Configure the standard local Release build/ tree\n"
	@printf "  local-bootstrap-debug  Configure the standard local Debug build/ tree\n"

local-bootstrap:
	tools/macos/configure-build-dir.sh

local-bootstrap-debug:
	BUILD_TYPE=Debug tools/macos/configure-build-dir.sh
