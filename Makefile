# Convenience targets.
#
# The important one is `make serve`. Cargo writes both feature variants to the
# same path, so a plain `cargo test` (which builds without features) silently
# replaces a CUDA-enabled server binary with one that cannot use the GPU. The
# symptom is a confusing "requires the cuda cargo feature" refusal at the next
# session. Always launching through these targets avoids it.

ORT ?= /usr/lib/libonnxruntime.so
MODEL ?= $(HOME)/.local/share/parakeet-dictation/nemotron
CONFIG ?= config.toml

.PHONY: test lint build-cuda serve dictate probe check

## Run the full suite. No GPU needed -- the mock backend covers the protocol.
test:
	cargo test

lint:
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --features cuda -- -D warnings

## Build the server WITH GPU support. Do this before `serve`.
build-cuda:
	ORT_DYLIB_PATH=$(ORT) cargo build -p parakeet-server --features cuda

## Run the server. Always rebuilds with cuda first, so a prior `make test`
## cannot leave a non-GPU binary in place.
serve: build-cuda
	ORT_DYLIB_PATH=$(ORT) ./target/debug/parakeet-server $(CONFIG)

## Run the dictation client.
dictate:
	cargo build -p parakeet-type
	./target/debug/parakeet-type start

## Report which execution provider is actually in use, and how fast.
probe: build-cuda
	PARAKEET_MODEL_DIR=$(MODEL) ORT_DYLIB_PATH=$(ORT) \
		cargo run -p parakeet-server --features cuda --example gpu_probe

## Golden-audio tests against the real model.
check: build-cuda
	PARAKEET_MODEL_DIR=$(MODEL) ORT_DYLIB_PATH=$(ORT) \
		cargo test -p parakeet-server --features cuda --test golden -- --ignored --nocapture
