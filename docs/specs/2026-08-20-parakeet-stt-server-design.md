# Parakeet STT Server — Design

**Date:** 2026-08-20
**Status:** Approved, pending implementation plan

## Purpose

A self-hosted streaming speech-to-text service, built on NVIDIA Parakeet/Nemotron
cache-aware streaming ASR via [`parakeet-rs`](https://github.com/altunenes/parakeet-rs).

It replaces two things that each solve half the problem:

- **whisrs** (whisper large-v3-turbo) — accurate, but whisper is not a streaming
  model, so text only lands at speech pauses.
- **nerd-dictation + Vosk** — genuinely word-by-word, but ~5.7% WER, no
  punctuation, and it hallucinates a filler `"the"` from room tone.

Parakeet is a transducer: it emits tokens as audio arrives, with constant memory
and no need to see a complete utterance. It reaches ~1.93% WER on librispeech
test-clean with punctuation — roughly 3x better than Vosk on the same benchmark,
in the same league as the whisper model, while still streaming.

### Goals

1. Transcribe audio from several sources — Linux desktop and Windows laptop,
   microphone or system audio.
2. **Live mode**: near-real-time transcription typed at the cursor.
3. **Transcript mode**: a GUI showing a running transcript the user can save.
4. **Bulk mode**: offline transcription of existing audio files.
5. Run locally now; move to a headless Ubuntu server (`acdc`) later with no
   protocol changes.

### Non-goals (v1)

- Server-side transcript storage or history. Clients own their files.
- Speaker diarization, translation, multilingual. English only.
- Session resume across reconnects.
- TLS. The service is LAN-only behind a shared token.

## Hard constraints discovered during design

These were measured or verified, not assumed. They drive most of the decisions
below.

### The deployment target caps CUDA at 12.8

`acdc` runs driver **570.211.01**, which supports **CUDA 12.8 maximum**. CUDA 13
requires driver >= 580, and CUDA forward-compatibility packages are a datacenter
feature that does not apply to GeForce cards. **A CUDA 13 image cannot run on
that host.**

The development desktop runs CUDA 13.3 with driver 610. Because CUDA is backward
compatible, a single **CUDA 12.8** container image runs on both machines. This is
the primary argument for containerizing: the host distributions differ (Arch vs
Ubuntu) and their CUDA versions differ, but the container fixes one stack for
both.

The RTX 2070 Super is compute capability **7.5** (Turing). CUDA 13 dropped
everything below 7.5, so Turing sits exactly on the boundary and remains
supported — one GPU generation older and this hardware would be unusable with
current toolkits.

### VRAM is contended by two services that fail visibly

`acdc`'s 2070 Super has **8192 MiB** total and is shared with **Frigate** (live
camera recording) and **Jellyfin** (NVENC transcoding). The nvidia-smi snapshot
taken during design showed ~580 MiB in use by `frigate.detector:onnx` plus two
ffmpeg processes — but that is a quiet-moment reading, not a ceiling.

Jellyfin's usage is **bursty and user-visible**: each concurrent transcode costs
roughly 150–400 MB, more for 4K HDR tonemapping. A footprint that is harmless at
03:00 can break someone's playback at 20:00. Both neighbours fail in ways a
person immediately notices — cameras stop recording, a film stops playing.

Measured model sizes (from the HuggingFace repo, not estimated):

| Model | Size | Streams | Punctuation |
|---|---|---|---|
| Nemotron streaming en 0.6B (fp32) | **2515 MB** | yes, 560 ms chunks | yes |
| EOU 120M streaming (fp32) | **481 MB** | yes, 160 ms chunks | **no** |
| TDT offline (fp32) | 2477 MB | no | yes |
| **TDT offline int8** | **670 MB** | no | yes |

Two consequences. Bulk is largely solved: **int8 TDT is 670 MB rather than
2477 MB**, a 3.7x reduction, so bulk jobs stop being the VRAM problem. But the
streaming model has **no int8 build published**, so **2515 MB is the floor** for
punctuated live transcription.

#### Tenancy policy

The governing rule: **this server is the lowest-priority GPU tenant on the host
and must actively yield.** It must never be the process that wins an allocation
race against the cameras or a film.

- **Zero footprint when idle.** Models are lazy-loaded on first use and
  **unloaded after an idle timeout**. The server holds no VRAM for most of the
  day. A keep-warm window prevents reload thrash during a dictation session.
- **Pre-load free-VRAM check.** Query free VRAM before loading. If there is not
  room for the model plus a safety margin, do not load — refuse the session with
  `error{code:"capacity"}`. Refusing is correct behaviour here, not a failure.
- **Explicit ORT arena cap**, so the server fails its own allocation rather than
  succeeding at a neighbour's expense.
- **Admission control** derived from measured inference time, refusing sessions
  rather than degrading everyone already connected.
- **Bulk uses int8 TDT** and yields to streaming sessions, which take priority.

#### CPU fallback is not a real fallback on this host

`acdc` runs a **Ryzen 1700** (Zen 1, 8c/16t, 2017). Zen 1 splits 256-bit AVX2
operations into two 128-bit halves, so it is roughly half-rate on exactly the
vector math inference depends on. A 0.6B encoder is very unlikely to sustain
real-time streaming there.

So "fall back to CPU" must not be written into the design as though it were a
graceful degradation path for the streaming model. Realistic options when the GPU
is unavailable are, in order of preference:

1. **Refuse the session** with a clear `capacity` error. Honest and predictable.
2. **Degrade to the EOU 120M model**, which is 5x smaller and may keep up on
   CPU — at the cost of punctuation.

Actual CPU inference time must be measured before option 2 is relied upon.

That Frigate already runs ONNX on this GPU is useful evidence: ORT + CUDA +
Docker GPU passthrough is proven on this exact hardware.

### The ASR model cannot revise its output

`Nemotron::transcribe_chunk` returns **only newly emitted tokens**, joined into a
string. A transducer emits a token and it is final; there is no mechanism by
which a later chunk rewrites an earlier one. Verified by reading
`parakeet-rs/src/nemotron.rs`.

This matters because the original request assumed the server would revise its
hypothesis and instruct the client to delete characters. **With this model it
never will.** Revision can only originate from an optional post-processing layer
above the ASR (punctuation refinement, LLM cleanup).

The protocol still defines `transcript.revise`, because retrofitting retraction
into a live protocol later is painful and post-processing is a plausible v2
feature. But **no v1 code path emits it**, and that should not be mistaken for a
bug.

### Concurrency is bounded by one mutex

`NemotronHandle` holds `Arc<Mutex<NemotronModel>>`. One ONNX session is shared
across all streams; `Nemotron::from_shared(&handle)` gives each stream
independent decoder state. This is what makes multi-client viable on 8 GB — N
clients share one ~2.4 GB model rather than loading N copies.

The cost is that **all inference is globally serialized**. Capacity is
`chunk_ms / inference_ms`; at a 560 ms chunk and a hypothetical 40 ms inference
that is ~14 concurrent real-time streams. **This number is an estimate and must
be measured** (see Testing). Admission control derives its limit from the
measured value.

## Architecture

A single Rust binary, shipped as a CUDA 12.8 container, with the model
volume-mounted.

```
parakeet-stt/
  crates/
    parakeet-proto/    wire types + serde. One definition of the protocol.
    parakeet-server/   axum + tokio + parakeet-rs. Ships in the container.
    parakeet-gui/      egui + cpal. Desktop + laptop, mic or system audio.
    parakeet-type/     headless Linux typer -> wtype. Replaces nerd-dictation.
```

`parakeet-proto` exists so clients and server cannot drift: both compile against
the same types, and a protocol change that breaks a client fails the build rather
than failing at runtime on the far side of a network.

### Why a single binary rather than split services

A split API-gateway/inference-worker design would allow restarting the protocol
layer without paying the model load, and could later shard across GPUs. Both are
real benefits, but they are worth roughly one inconvenience a month against
permanent extra moving parts and an IPC serialization hop on every 560 ms chunk.

The protocol is identical either way, so if model-reload pain becomes real, the
binary can be split later without touching a single client.

### Container

```
nvidia/cuda:12.8-cudnn-runtime  (base)
  + libonnxruntime (official CUDA 12 build)
  + /app/parakeet-server
  + /models   <- volume mount, NOT baked into the image
```

Model as a volume keeps the image ~1 GB instead of ~4 GB and allows swapping
models without a rebuild. Multi-stage build so the Rust toolchain does not ship.
Requires NVIDIA Container Toolkit on the host, already present on `acdc`.

## Protocol

WebSocket at `/v1/stream`. JSON text frames carry control; **binary frames carry
raw PCM**, avoiding base64's 33% overhead on the high-rate path.

Auth is a shared bearer token in the handshake. The server binds to the LAN. No
TLS — this is a trusted home network, and cert distribution to a Windows laptop
is friction without a matching threat.

### Client to server

```jsonc
{"type":"session.start","mode":"live"|"transcript",
 "sample_rate":16000,"encoding":"pcm_s16le",
 "language":"en","vocabulary":["Sway","PipeWire"]}   // vocabulary optional

<binary frames: raw PCM>

{"type":"session.flush"}   // force emission of buffered audio
{"type":"session.stop"}
```

### Server to client

```jsonc
{"type":"session.ready","session_id":"...","chunk_ms":560,"model":"nemotron-en-0.6b"}

{"type":"transcript.commit","seq":1,"text":"the timeout is thirty seconds"} // final, both modes
{"type":"transcript.provisional","seq":2,"text":"and the retry"}      // transcript mode only
{"type":"transcript.revise","seq":2,"retract_n":13,"text":"and the retries"} // reserved, see above

{"type":"error","code":"capacity","message":"...","retryable":true}
{"type":"session.closed","reason":"..."}
```

### Mode semantics

This is the core safety property of the design.

- **live** — emits `transcript.commit` **only**. Append-only. The client is
  typing into arbitrary applications, where deleting characters is destructive if
  the user has typed or moved the cursor since. Costs a little latency; can never
  eat the user's work.
- **transcript** — may additionally emit `provisional` and `revise`. The client
  owns the buffer being edited, so rewriting it is safe.

One session flag selects between them. Same protocol either way.

### Bulk jobs

Plain HTTP, since these are not latency-sensitive and can run for a long time:

```
POST /v1/jobs        (multipart audio file)  -> {"job_id":"...","status":"queued"}
GET  /v1/jobs/{id}                           -> {"status":"running","pct":42}
                                             -> {"status":"done","text":"..."}
```

Bulk uses the **non-streaming Parakeet TDT** model, which is more accurate than
the streaming one because it sees full context. Jobs are serialized on the GPU
behind streaming sessions, which take priority — a live dictation must not stall
behind an hour-long file.

Job results are held **in memory with a TTL**. This is the only server-side
state, and it is a work queue rather than a transcript archive, consistent with
the decision that clients own storage.

## Clients

| Client | Platform | Mode | Purpose |
|---|---|---|---|
| `parakeet-gui` | Linux + Windows | transcript | Running transcript, save to file, choose mic or system audio |
| `parakeet-type` | Linux/Sway | live | Types at the cursor via `wtype` |

**egui/eframe + cpal** for the GUI: a single self-contained binary on both
Wayland and Windows with no system UI dependencies. cpal covers capture on both
platforms and supports **WASAPI loopback** on Windows, which is what makes
"system audio from the laptop" work; PipeWire monitor sources give the equivalent
on Linux. One codebase covers desktop and laptop, mic and system audio.

`parakeet-type` has no UI and stays deliberately tiny. It is what replaces
nerd-dictation.

**v1 scope is server + `parakeet-type`.** It gets off Vosk soonest and proves the
protocol against the simpler client. The GUI follows.

## Error handling

- **Backpressure**: each session has a bounded audio queue. When a client
  outruns the server, the server reports it explicitly rather than buffering
  without limit and drifting further behind real time.
- **Admission control**: sessions beyond the measured concurrency limit are
  refused with `error{code:"capacity", retryable:true}`, not accepted into a
  degraded pool.
- **Auth failure / malformed frames**: `error` followed by close.
- **VRAM exhaustion**: caught at model load and session start; reported as
  `capacity`. Never allowed to propagate as a GPU OOM that could disturb other
  workloads on the host.
- **Reconnect**: clients reconnect and start a fresh session. No resume in v1 —
  it is meaningful complexity for a case that barely matters at dictation
  durations.

## Testing

- **Protocol conformance without a GPU.** A mock ASR backend behind the same
  trait lets the entire session lifecycle, mode semantics, backpressure, and
  error paths be tested in CI on any machine. This is most of the risk surface
  and none of it needs CUDA.
- **Golden audio fixtures.** Known WAV files with expected transcripts, to catch
  quality regressions from model or preprocessing changes.
- **Concurrency soak.** N synthetic streams driven at real-time rate to find the
  actual ceiling and validate that admission control refuses cleanly rather than
  degrading everyone. This measurement replaces the ~14-stream estimate.
- **VRAM ceiling test.** Verify the arena cap holds and that a bulk job cannot
  push total usage into Frigate's headroom.

## Open questions

- Measured inference time per 560 ms chunk on both a 3060 Ti and a 2070 Super,
  which sets the real concurrency limit.
- Measured **model load and unload time**, which sets the keep-warm window and
  determines how much latency the idle-unload policy costs on first use.
- Measured **CPU** inference time on Zen 1, to establish whether the EOU
  degradation path is real or whether refusing is the only honest option.
- Whether the official ONNX Runtime CUDA 12 build is packaged conveniently for
  the container, or whether it needs building from source in the image.
- Whether quantizing the streaming Nemotron to int8 ourselves is viable. No int8
  build is published for it, and streaming models carry cache tensors that can be
  awkward to quantize, but a ~4x cut to ~650 MB would materially change the
  tenancy story on a contended GPU.
