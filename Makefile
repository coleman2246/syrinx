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

# The iOS build runs on a macOS VM: Xcode will not run anywhere else, and the
# ring crate compiles C and assembly that needs the iOS SDK, so Linux can
# `cargo check` those targets but cannot link them. See docs/ios.md.
MACVM ?= ssh -i $(HOME)/.ssh/macvm -o StrictHostKeyChecking=no -p 2222 cole@127.0.0.1
IPA ?= $(HOME)/Downloads/SyrinxDemo.ipa

.PHONY: test lint build-cuda serve dictate probe check ios ios-framework

## Run the full suite. No GPU needed -- the mock backend covers the protocol.
##
## The second line is the diarizer's own tests. They need no models and take
## well under a second, but they are behind the feature, so a plain `cargo
## test` runs 66 of syrinx-server's tests and skips 20 -- everything covering
## the VAD and embedding wrappers and the model-resolution rules.
test:
	cargo test
	cargo test --features diarize --lib

## Clippy over every feature combination that ships. The last two exist
## because `examples/diarize_probe` is `required-features = ["diarize"]`, so
## without them cargo skips it entirely and nothing in this repo ever
## type-checks the harness: changing `OnlineClusterer::with_params` or
## deleting one of its diagnostics would leave `make lint` and `make test`
## both green and the probe broken.
lint:
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --features cuda -- -D warnings
	cargo clippy --all-targets --features diarize -- -D warnings
	cargo clippy --all-targets --features cuda,diarize -- -D warnings

## Build the server WITH GPU support. Do this before `serve`.
build-cuda:
	ORT_DYLIB_PATH=$(ORT) cargo build -p syrinx-server --features cuda

## Run the server. Always rebuilds with cuda first, so a prior `make test`
## cannot leave a non-GPU binary in place.
serve: build-cuda
	ORT_DYLIB_PATH=$(ORT) ./target/debug/syrinx-server $(CONFIG)

## Run the dictation client (types at the cursor).
dictate:
	cargo build -p syrinx-cli
	./target/debug/syrinx start --mode type

## Run the GUI.
gui:
	cargo build -p syrinx-gui
	./target/debug/syrinx-gui

## Report which execution provider is actually in use, and how fast.
probe: build-cuda
	SYRINX_MODEL_DIR=$(MODEL) ORT_DYLIB_PATH=$(ORT) \
		cargo run -p syrinx-server --features cuda --example gpu_probe

## Golden-audio tests against the real model.
check: build-cuda
	SYRINX_MODEL_DIR=$(MODEL) ORT_DYLIB_PATH=$(ORT) \
		cargo test -p syrinx-server --features cuda --test golden -- --ignored --nocapture

## Build the iPhone app on the macOS VM and copy the .ipa back.
##
## The VM builds from its own clone, so this pushes first -- otherwise it
## cheerfully builds the last thing you committed and the change you are
## testing is not in it.
ios:
	git push -q
	$(MACVM) 'cd ~/syrinx && git pull -q && ./ios/generate.sh && ./ios/build-ipa.sh'
	scp -i $(HOME)/.ssh/macvm -o StrictHostKeyChecking=no -P 2222 \
		cole@127.0.0.1:syrinx/ios/build/SyrinxDemo.ipa $(IPA)
	@echo "wrote $(IPA)"

## Rebuild the Rust core for iOS. Needed whenever anything under crates/
## changed: `make ios` links the existing static library and will otherwise
## produce an app without your change, silently.
ios-framework:
	git push -q
	$(MACVM) 'cd ~/syrinx && git pull -q && ./crates/syrinx-ios/build-xcframework.sh'
