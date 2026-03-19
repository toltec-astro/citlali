.PHONY: help local-bootstrap local-bootstrap-debug

help:
	@printf "Available targets:\n"
	@printf "  local-bootstrap        Configure the standard local Release build/ tree\n"
	@printf "  local-bootstrap-debug  Configure the standard local Debug build/ tree\n"

local-bootstrap:
	tools/local/configure-build-dir.sh

local-bootstrap-debug:
	BUILD_TYPE=Debug tools/local/configure-build-dir.sh
