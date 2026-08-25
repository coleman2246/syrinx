# syrinx-server, for a GPU host.
#
# CUDA 12.8 is a constraint, not a preference. The deployment host runs driver
# 570, which caps at 12.8, and GeForce cards get no forward-compatibility
# package -- a CUDA 13 image cannot start there. ONNX Runtime 1.28 publishes
# both a cuda12 and a cuda13 build and defaults to 13, so the cuda12 asset is
# named explicitly below.
#
# Models are volume-mounted, never copied in. They are 2.5 GB and change on a
# different schedule from the code; baking them in would mean rebuilding and
# repulling gigabytes to change a line of Rust.
#
# `diarize` is compiled in alongside `cuda` so speaker labelling is a matter of
# mounting 29 MB of models and naming them in the config, not of rebuilding the
# image. It costs nothing when unconfigured -- no model is loaded and no session
# is labelled -- and it adds no runtime dependency, since it drives the same
# ONNX Runtime the ASR already links.

# ---------------------------------------------------------------- build stage
# ort 2.0.0-rc.13 requires Rust 1.88.
FROM rust:1.88-bookworm AS build

WORKDIR /src

# Manifests first, so the dependency build is cached until they change. Every
# member needs a placeholder source file or cargo refuses to plan the build.
COPY Cargo.toml Cargo.lock ./
COPY crates/syrinx-proto/Cargo.toml crates/syrinx-proto/
COPY crates/syrinx-audio/Cargo.toml crates/syrinx-audio/
COPY crates/syrinx-client/Cargo.toml crates/syrinx-client/
COPY crates/syrinx-server/Cargo.toml crates/syrinx-server/
COPY crates/syrinx-cli/Cargo.toml crates/syrinx-cli/
COPY crates/syrinx-gui/Cargo.toml crates/syrinx-gui/
RUN mkdir -p crates/syrinx-proto/src crates/syrinx-audio/src crates/syrinx-client/src \
             crates/syrinx-server/src crates/syrinx-cli/src crates/syrinx-gui/src \
 && touch crates/syrinx-proto/src/lib.rs crates/syrinx-audio/src/lib.rs \
          crates/syrinx-client/src/lib.rs \
 && echo 'fn main() {}' > crates/syrinx-server/src/main.rs \
 && echo 'fn main() {}' > crates/syrinx-cli/src/main.rs \
 && echo 'fn main() {}' > crates/syrinx-gui/src/main.rs \
 && cargo build --release -p syrinx-server --features cuda,diarize \
 && rm -rf crates

# Now the real sources. Only syrinx-server and what it depends on: the GUI
# needs a windowing stack this image has no reason to carry.
COPY crates/syrinx-proto crates/syrinx-proto
COPY crates/syrinx-server crates/syrinx-server
COPY crates/syrinx-audio crates/syrinx-audio
COPY crates/syrinx-client crates/syrinx-client
COPY crates/syrinx-cli crates/syrinx-cli
COPY crates/syrinx-gui crates/syrinx-gui
# Touch so cargo does not trust the placeholder timestamps.
RUN touch crates/*/src/lib.rs crates/*/src/main.rs 2>/dev/null; \
    cargo build --release -p syrinx-server --features cuda,diarize

# -------------------------------------------------------------- runtime stage
# `cudnn` in the tag matters: ONNX Runtime's CUDA provider needs cuDNN 9, and
# the plain runtime image does not carry it.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG ORT_VERSION=1.28.0
ARG DEBIAN_FRONTEND=noninteractive

# ca-certificates for the download; curl removed again afterwards so the final
# image carries no fetch tooling.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSL -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu_cuda12-${ORT_VERSION}.tgz" \
 && mkdir -p /opt/onnxruntime \
 && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
 && rm /tmp/ort.tgz \
 && apt-get purge -y curl \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# ort links ONNX Runtime at run time, so the container chooses the build rather
# than cargo downloading one compiled against the wrong CUDA.
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
# The CUDA execution provider is dlopened from beside it.
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib

# Everything the driver is asked for. `video` is what exposes the GPU when the
# runtime is not injecting the full set.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=build /src/target/release/syrinx-server /usr/local/bin/syrinx-server
COPY docker/config.toml /etc/syrinx/config.toml

# Unprivileged: the service reads a model directory and answers a socket, and
# needs nothing else. Not `nobody`, which is shared with every other unowned
# process in the image.
RUN useradd --system --no-create-home --uid 10001 syrinx
USER syrinx

EXPOSE 8770
ENTRYPOINT ["/usr/local/bin/syrinx-server", "/etc/syrinx/config.toml"]
