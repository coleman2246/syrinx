# Syrinx — Design

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
5. Run locally now; move to a headless Ubuntu GPU server later with no
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

the GPU host runs driver **570.211.01**, which supports **CUDA 12.8 maximum**. CUDA 13
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

the GPU host's 2070 Super has **8192 MiB** total and is shared with **Frigate** (live
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

- **Near-zero footprint when idle.** Models are lazy-loaded on first use and
  **unloaded after an idle timeout**. Measured: 0 MiB before the first session,
  3398 MiB while serving, **164 MiB once unloaded** -- a 95% reduction. The
  residual is the CUDA context, which cannot be released without exiting the
  process, so "zero when idle" is not literally achievable. A keep-warm window
  prevents reload thrash during a dictation session.
- **Pre-load free-VRAM check.** Query free VRAM before loading. If there is not
  room for the model plus a safety margin, do not load — refuse the session with
  `error{code:"capacity"}`. Refusing is correct behaviour here, not a failure.
- **Explicit ORT arena cap**, so the server fails its own allocation rather than
  succeeding at a neighbour's expense.
- **Admission control** derived from measured inference time, refusing sessions
  rather than degrading everyone already connected.
- **Bulk uses int8 TDT** and yields to streaming sessions, which take priority.

#### Measured performance (2026-08-20, RTX 3060 Ti + Ryzen 5700X)

Steady-state medians over 20 chunks, warmup discarded:

| Provider | Per 560 ms chunk | RTF | Concurrent streams | VRAM |
|---|---|---|---|---|
| CUDA (3060 Ti) | **37 ms** | 0.07x | ~15 | **3400 MiB** |
| CPU (Ryzen 5700X) | **149 ms** | 0.27x | ~4 | 0 |

Two corrections to earlier estimates in this document:

- **VRAM is 3400 MiB, not ~2515 MiB.** The model file is 2515 MB, but ORT's CUDA
  arena and workspace add ~900 MB on top. The tenancy budget must use the larger
  figure.
- The first chunk costs **~280 ms** on GPU against a 37 ms steady state, because
  it pays for CUDA context creation and cuDNN autotuning. Quoting first-chunk
  timing understates GPU performance by nearly an order of magnitude, and
  flatters CPU, which has no equivalent warmup.

The ~15 concurrent streams figure confirms the earlier ~14 estimate.

#### ONNX Runtime's CUDA provider does not link cuDNN

`libonnxruntime_providers_cuda.so` references 8 cuDNN symbols
(`cudnnGetConvolutionBackwardDataAlgorithm_v7` and friends) but carries **no
`DT_NEEDED` entry for cuDNN** -- confirmed with `objdump -p`. It links cublas,
cudart and nccl; the cuDNN link was dropped, most likely by `--as-needed` at
package build time. When ORT `dlopen`s the provider those symbols are
unresolved, registration fails, and **ORT falls back to CPU silently**.

This was diagnosed only because the plan required verifying with `nvidia-smi`
rather than trusting a passing test. The golden tests were green the whole time,
transcribing correctly at a quarter of the intended speed.

The fix is `dlopen`ing cuDNN with `RTLD_GLOBAL` at startup, in
`asr::parakeet::preload_cudnn`. It is equivalent to
`LD_PRELOAD=/usr/lib/libcudnn.so.9` but done in-process, so deployment does not
depend on an environment variable that is easy to omit.

`ParakeetBackend::verify_gpu` then fails startup if inference runs slower than
90 ms per chunk, turning an invisible 4x performance cliff into an explicit
error. It is timing-based rather than provider-querying deliberately: it
measures the property actually cared about, and catches every cause rather than
only this one known failure.

#### CPU is a viable fallback after all

An earlier draft of this document asserted that CPU "is not a real fallback",
reasoning from Zen 1's halved AVX2 throughput without measuring. At 149 ms per
560 ms chunk on Zen 3, that was too pessimistic: CPU keeps up in real time with
substantial margin, supporting ~4 concurrent streams.

Zen 1 on the GPU host will be slower -- perhaps 2-3x -- which still plausibly lands
under the 560 ms real-time budget for a single stream, though not for several.
So CPU-only operation on the deployment host is a genuine option that would
sidestep GPU contention with Frigate and Jellyfin entirely. It must be measured
there before being relied upon.

#### The deployment host's CPU

the GPU host runs a **Ryzen 1700** (Zen 1, 8c/16t, 2017). Zen 1 splits 256-bit AVX2
operations into two 128-bit halves, so it is roughly half-rate on exactly the
vector math inference depends on. An earlier draft read that as ruling CPU out
entirely; the measurements above show it was too pessimistic.

Extrapolating 149 ms on Zen 3 by 2-3x puts Zen 1 somewhere around 300-450 ms per
560 ms chunk. That still fits the real-time budget for **one** stream, with
little room for a second. So when the GPU is unavailable, the options are, in
order of preference:

1. **Run on CPU with a reduced session limit.** Viable if measurement on the GPU host
   confirms the extrapolation, and it sidesteps GPU contention entirely.
2. **Refuse the session** with a clear `capacity` error. Honest and predictable.
3. **Degrade to the EOU 120M model**, 5x smaller and comfortably real-time on
   CPU — at the cost of punctuation.

The extrapolation must be measured on the GPU host before option 1 is relied upon.

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
syrinx/
  crates/
    syrinx-proto/    wire types + serde. One definition of the protocol.
    syrinx-server/   axum + tokio + parakeet-rs. Ships in the container.
    syrinx-gui/      egui + cpal. Desktop + laptop, mic or system audio.
    parakeet-type/     headless Linux typer -> wtype. Replaces nerd-dictation.
```

`syrinx-proto` exists so clients and server cannot drift: both compile against
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
  + /app/syrinx-server
  + /models   <- volume mount, NOT baked into the image
```

Model as a volume keeps the image ~1 GB instead of ~4 GB and allows swapping
models without a rebuild. Multi-stage build so the Rust toolchain does not ship.
Requires NVIDIA Container Toolkit on the host, already present there.

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
| `syrinx-gui` | Linux + Windows | transcript | Running transcript, save to file, choose mic or system audio |
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

## Windows portability (audit, 2026-08-21)

> **Superseded by the section below.** The audit was written before any Windows
> testing; the port has since been built and run. Kept because comparing the two
> shows which predictions held. Most did. Two did not, and both were assumptions
> stated as fact.

### Compiles and works

- `syrinx-proto`, `syrinx-audio::meter`, `syrinx-client::{mode, save, bulk}` are
  all portable.
- The cpal backend enumerates microphones and offers output devices as system
  audio, which WASAPI turns into loopback.
- File transcription: ffmpeg exists on Windows and the streaming path is
  platform-neutral.

### Will not compile

| Component | Reason |
|---|---|
| `ipc.rs`, `daemon.rs` | `UnixStream` / `UnixListener` |
| `state.rs` | `nix` for `kill(pid, 0)` and SIGTERM |
| `cli/main.rs` | `tokio::signal::unix` |

Named pipes are the Windows equivalent of a Unix socket, and a named event or a
console control handler replaces SIGTERM. The shape of the code does not change,
only the transport, so an `ipc::transport` module with two implementations would
cover it.

### Compiles but does nothing useful

- `wtype` is Wayland-only, so **typing at the cursor does not work**. Windows
  needs `SendInput`, which is a different mechanism, not a different binary.
- `pw-dump` / `pw-record` / `pw-link` / `pactl` are PipeWire, so per-application
  capture is unavailable. Windows has process loopback since 10 2004, but cpal
  does not expose it.
- `pkill -RTMIN+N waybar` is meaningless off Linux, and harmless: the command
  simply fails.
- `nvidia-smi` for the VRAM guard is present on Windows with an NVIDIA driver,
  so that one is fine.
- `date` for timestamps is not a Windows command. It falls back to an epoch
  stamp, so saving still works but filenames are uglier. Worth replacing with a
  small time formatter.

### Summary (as predicted, before testing)

The **server** is portable in principle but is meant to run in a container on
Linux, so this does not matter. The **GUI and CLI need an IPC transport
abstraction and a Windows text-injection path** before they build, and
per-application capture will remain Linux-only. Transcribe mode, file
transcription, the meter and system audio would all work.

## Bulk throughput (measured 2026-08-21, RTX 3060 Ti)

113.2 s of audio, warm server:

| | Wall clock | Throughput |
|---|---|---|
| Cold (model loading + GPU verify) | 10.0 s | 11x |
| Warm | 7.6 s | **15x** |

**This is the ceiling, not a tuning failure.** 113.2 s is 202 chunks of 560 ms,
and inference measures 37 ms per chunk, so the model alone needs 7.5 s. Measured
wall clock is 7.6 s: about **99% efficiency**, with essentially no protocol or
transport overhead left to remove. An hour of audio takes roughly four minutes.

An earlier figure of "2x" was misleading and came from a 6.8 s test file, where
~2.3 s of cold start dominated everything else.

### Why there is no reject-and-retry

Sending flat out and having the server discard chunks it cannot keep up with
would be worse than what is there. Discarded audio is either lost from the
transcript or has to be re-sent, and re-sending work the server already
half-did wastes the very capacity being contended for.

Flow control achieves the same goal without either problem. The server's audio
queue is bounded, so a full queue stops the reader draining the socket, TCP
stops acknowledging, and the client blocks in `send` exactly as long as needed.
The client already sends as fast as the server can consume, and no audio is
dropped or repeated. The 99% figure is what that looks like when it works.

### Making it genuinely faster

Not achievable at the protocol layer, since the protocol is not the constraint.
The real options:

- **A faster model.** int8 TDT is roughly 3.7x smaller and built for offline
  decoding, where it sees full context rather than a streaming window.
- **Batched inference**, feeding several chunks per forward pass. A model and
  runtime capability, not something the server can arrange.
- **Concurrent sessions do not help.** `NemotronHandle` holds
  `Arc<Mutex<NemotronModel>>`, so every session serialises on one model. Two
  short concurrent sessions complete correctly, but three long ones did not
  finish inside ten minutes, which wants investigating before `max_sessions`
  above 1 is trusted for bulk work.


## Windows port (measured, 2026-08-21)

Built and run on Windows 11 Pro x64, MSVC toolchain, client on the LAN against
the CUDA server on the development desktop. 148 tests pass on Windows.

**Verified working:** source enumeration, first-run config creation under
`%APPDATA%`, file transcription over the LAN, live transcription of system audio
via WASAPI loopback, the daemon and its named-pipe IPC, and stopping a session
and the daemon from the CLI.

### What the audit got wrong

**cpal loopback is only half transparent.** The audit said WASAPI "turns" an
output device into loopback, and the cpal backend's own module doc said the same
— I had written both from cpal's crate documentation, which says using an output
device as an input "will transparently enable loopback mode". It does set
`AUDCLNT_STREAMFLAGS_LOOPBACK` when *building* an input stream on a render
endpoint. But `default_input_config()` refuses to describe one, answering
"Device does not support input", so opening a system-audio source failed outright.
The fix is one branch: take the config from `default_output_config()` and open it
for input. Measured with a 440 Hz tone, the meter lights a single band at 60%.

**A stop mechanism, not just a signal.** The audit expected "a named event or a
console control handler" to replace SIGTERM. Both are worse than they look: a
console control handler only reaches a process with a console, and the daemon has
none. What shipped is a stop-request file beside the PID file, polled every 200 ms
by the running session. `TerminateProcess` was rejected outright: it gives the
session no chance to flush the last utterance, which is the one thing dictation
must not lose.

### What the audit missed

**A daemon with no tray could not be stopped.** `stop`, `toggle` and `status`
only ever consulted the PID file, which a daemon does not write — so `status`
reported "idle" while the GUI was dictating, and `stop` was a no-op. On Linux the
tray hid this. Windows has no tray, so the daemon could only be killed. All three
commands now fall through to the daemon over IPC, and `quit` was added. This was
a real Linux bug too, and directly against the stated goal that the CLI be
equivalent to the GUI.

**A PID file is not enough on Windows.** `OpenProcess` succeeding does not mean
the process is alive: a handle stays valid for a process that has exited but not
been reaped. The exit code has to be checked as well, or a crashed session reads
as running forever and the toggle never starts again — the same failure the PID
file exists to prevent.

### Still Linux-only

Per-application capture (PipeWire) and the system tray (ksni is DBus-based).
Neither blocks dictation; the tray gap is covered by `syrinx quit`.

### Not verified

Typing at the cursor. `SendInput` is implemented and its encoding is unit-tested,
but Windows refuses synthetic input from a non-interactive window station, so it
cannot be exercised over SSH: the integration test reports 0 of 34 events
delivered. It needs a run from a real login. The error message names elevation as
the likely cause, which is the other common reason for the same symptom.

The laptop microphone also reads silent, but privacy consent is `Allow` and the
endpoints report OK, so this is most likely a quiet room rather than a fault.


## Tray and hotkey (2026-08-21)

The Windows tray uses `tray-icon`; Linux keeps `ksni`. Both feed the same
`TrayCommand` channel, so the daemon does not know which is running. The icon on
Windows is generated in code -- a coloured disc, red while listening -- rather
than shipped as an asset: there is nothing to lose, nothing to keep in step with
the Linux icon names, and no build step.

`tray-icon` and `global-hotkey` are both thread-affine on Windows: each creates
a hidden window and receives its messages on the creating thread. They therefore
share one thread and one message pump. The daemon's own loop is a poll over
channels, not a message loop, so this could not live there.

### The hotkey is not portable, and cannot be

Windows and X11 let any process claim a key combination. **Wayland deliberately
does not** -- the compositor owns input, and a client that could grab keys
globally could keylog every other client. There is a desktop portal for this
(`org.freedesktop.portal.GlobalShortcuts`) but wlroots does not implement it, so
Sway has nothing to offer either.

The config setting is uniform anyway, and the daemon reports which case applies
at startup with the exact compositor line to add. A hotkey that silently does
nothing would be worse than one never offered.

Not verified: registration fails over SSH with "requires an interactive window
station" (error 1459), the same limitation that blocks `SendInput`. It needs a
run from a real login.

### A GUI bug found on the way

`ensure_daemon` spawned the daemon with stdout and stderr on `/dev/null` and
never waited on the child. Two consequences, both seen on the development
desktop: a daemon that failed to start reported only "did not start listening
within 5s" with the reason discarded, and the dead child stayed a zombie for as
long as the GUI was open. Output now goes to a log beside the PID file, the exit
is detected while polling and reported with the log's tail, and the child is
reaped in the background once the socket is up.


## One config for every platform (2026-08-21)

The generated config was platform-specific: it listed only the injection methods
the generating machine could use, and omitted `waybar_signal` off Linux. That
reasoning was "offering a Windows user `wtype` is a wrong answer dressed as a
choice", which holds right up until the same file is kept for a desktop and a
laptop. Then the file is wrong on whichever machine did not write it.

It is now byte-identical on both, verified by hashing the output on Linux and
Windows: `013031935437be...`. Every option is listed with the platform it
applies to, and nothing is hidden.

`inject` defaults to `auto` rather than to a concrete method, so the default
needs no platform of its own. It resolves once per process: `sendinput` on
Windows; on Linux `ydotool` if `ydotoold` is running, else `wtype`. Probing for
ydotoold rather than assuming means `auto` is also correct in Electron
applications, where `wtype` loses focus -- and running that daemon is
deliberate enough to read as consent. Cached in a `OnceLock` because it is
consulted once per transcript fragment, several times a second.


## A host instead of a URL (2026-08-21)

The client config asked for `url = "ws://192.168.1.10:8770/v1/stream"`: four
things to get right when only one of them ever changes. It is now
`server = "192.168.1.10"`, with the scheme, port and endpoint supplied.

Deliberately dull rules, because one that guesses is worse than one that is
boring: a bare host gets the default port and the endpoint appended; anything
written as a URL is respected as-is with only a missing path filled in, and no
port is invented -- a proxy on 443 is exactly the case the `url` override
exists for. A full URL remains accepted, so nothing that worked stops working.

An older config is migrated on load: where its URL is one syrinx would have
built anyway, the host is lifted out and the URL dropped. Anything unusual is
left exactly as written. A test asserts the migration never changes where a
client connects, which is the only real risk in it.

The default mode is now `type`, not `transcribe`.

### A trap this set

Deleting the canonical config does **not** regenerate it: `Config::load` falls
back to the pre-rename `~/.config/parakeet/config.toml`, which still exists on
the development desktop. That fallback was silent, so an old file quietly took
over -- and with the default now typing at the cursor rather than accumulating
a transcript, an unnoticed config is no longer a harmless surprise. Loading a
non-canonical path is now logged as a warning naming both paths.


## Streaming to a file, and a theme (2026-08-21)

### Streaming

Saving at the end is fine until something goes wrong at minute forty. Each
committed fragment is now appended as it arrives, and the file is opened for
append so a second session continues the first.

Three decisions worth recording:

**Commits only.** The server sends nothing else today -- the ASR is append-only
-- but a post-processing layer would emit provisional text that a revision
retracts, and a flushed file cannot take anything back. Writing commits only
means the file lags the screen slightly and is never wrong.

**No buffering.** `File` is unbuffered, so a write reaches the operating system
before the call returns and a process that dies afterwards loses nothing.
Surviving a power cut too would need an fsync per fragment, which is a real
cost for a rarer failure. Verified with `kill -9` mid-session: everything
spoken was on disk.

**Lines follow speech, not chunks.** The first implementation put every
fragment on its own stamped line, which broke words in half -- "brown fox j"
then "umps over the" -- because the model splits on 560 ms boundaries. A line
now continues while speech continues and breaks after 1.5 s of silence. Nothing
is buffered to achieve it; the newline is simply withheld.

In separate mode every session shares one file. `O_APPEND` takes the offset
atomically, so fragments interleave in arrival order rather than tearing, which
is what the labelled layout wants anyway.

### Theme

Colours were literals at every call site, which is how an interface drifts:
two reds that are nearly the same, a green that belongs to nothing else. They
are now one palette, taken from Fluent 2 -- the violet brand colour, near-neutral
greys, and Teams' own red and green for status. The transcript sits on a
surface of its own rather than on bare canvas, which was reading as an empty
window with text in the corner.


## Timestamps in filenames (2026-08-21)

`timestamp()` shelled out to `date +%Y-%m-%d_%H-%M-%S`, which is not a command
on Windows: every generated filename there fell back to `epoch-1755…`. The
audit predicted this and called it cosmetic; it is not, because a folder of
epoch counts cannot be read or ordered by eye.

Formatted in-process with `chrono` now -- one dependency, the same answer on
both platforms, and one fewer process spawn per save. Local time, ordered
largest unit first so that sorting by name sorts by time. The name is the
timestamp alone: `2026-08-21_14-59-41.txt`.

The **Stream…** dialog previously offered a fixed `transcript.txt`, which
quietly invited overwriting the last session's file.

### A repository bug found alongside it

`.gitignore` had `!tests/fixtures/golden/*.wav`, which without `**` only
matches at the repository root. The fixture lives under
`crates/syrinx-server/tests/`, so it was never tracked -- the golden test could
not run from a fresh clone, and the file never reached the Windows laptop,
whose tree arrives by `git archive`.


## Launching on Windows (2026-08-21)

Double-clicking the GUI has to start a daemon, open a window, and leave the
daemon running when the window closes -- including when the terminal that
launched it closes.

`windows_subsystem = "windows"` on release builds means no console appears.
Debug builds keep their console deliberately, so day-to-day use wants the
release binary.

The daemon is spawned with `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`. A
child otherwise inherits its parent's console and receives `CTRL_CLOSE_EVENT`
when it is closed, which would stop dictation whenever the launching terminal
went away. Verified with `AttachConsole` against the running daemon: error 6,
`ERROR_INVALID_HANDLE` -- it has no console at all, so no console event can
reach it.

The Unix side uses `setsid` in `pre_exec` for the same reason. A new process
group is not enough: a background job in the same session is still sent SIGHUP
when the shell exits. Confirmed the daemon is its own session leader.

Startup failures now open a message box. A double-clicked application has no
console to print to, so a bad token used to look exactly like a broken
download: no window, no message.

### Not verifiable over SSH

Windows OpenSSH puts a session's processes in a job object and terminates it
when the connection closes, which kills the daemon regardless of any creation
flag. Launching through `Win32_Process.Create` escapes that job, and the daemon
then survived across an entirely new SSH connection.


## Container (measured, 2026-08-21)

Built and run on the development desktop with GPU passthrough before going
anywhere near the deployment host.

**Verified:** the container process holds VRAM on the host GPU
(`/usr/local/bin/syrinx-server`, 3378 MiB), transcribes correctly through the
published port, unloads the model when idle, refuses to load when another
tenant leaves too little free, and answers its own healthcheck. Release build
in the image: 20.9 ms per chunk against 35 ms for the debug build on the host.

### Two numbers the plan got wrong

**Size: 5.12 GB, not the "roughly 1 GB" the plan predicted.** Models are
correctly absent -- that part of the check was sound -- but the estimate ignored
what the runtime costs. `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` is about
4.4 GB, and ONNX Runtime adds 622 MB of which `libonnxruntime_providers_cuda.so`
alone is 593 MB. Our own binary is 6.5 MB. There is no meaningful trimming to do
without hand-assembling the CUDA runtime, which trades gigabytes for a class of
missing-library failure that is expensive to diagnose remotely.

**"Zero VRAM when idle" is 144 MiB.** The model does unload -- 3378 MiB drops to
144 MiB -- but a CUDA context stays resident for the life of the process.
Reaching literal zero would mean exiting the process, not unloading the model.
144 MiB against a shared 8 GB card is not worth that.

### ONNX Runtime 1.28 defaults to CUDA 13

The release publishes `onnxruntime-linux-x64-gpu_cuda12-1.28.0.tgz` and
`..._cuda13-...`, and the release notes say the pipeline has moved to CUDA 13.
Taking the default would produce an image that cannot start on driver 570. The
Dockerfile names the cuda12 asset explicitly, and the version is an ARG so the
pairing stays visible.

### Secrets

The token comes from `SYRINX_TOKEN`. A token in a layer is published to anyone
who can pull the image; a token in a committed config is published to anyone
who can read the repository. `bind` and `model_dir` are overridable for the same
reason -- they differ per deployment. The behaviour settings (VRAM floor,
session cap, idle timeout) deliberately are not: they describe how the service
behaves under pressure and belong in a file that can be reviewed.

An empty environment variable does not override. An unset variable in a compose
file arrives as `""`, and taking that literally would replace a working token
with one that fails closed -- a failure that looks like a client problem.


## TLS (2026-08-21)

The design assumed a LAN and said so: a shared bearer token over plaintext
`ws://`, with the reasoning that TLS would add cert-distribution friction on a
Windows laptop for no gain on a trusted network. That reasoning held until the
laptop needed to reach the service from elsewhere, and a VPN turned out not to
be available -- the machine already runs Tailscale for something else, and a
device can only be on one tailnet at a time.

Two pieces were missing, and only one of them was code.

**The client had no TLS backend.** The address parser accepted `wss://` and had
done since the host-not-URL change, but `tokio-tungstenite`'s default features
are `connect` and `handshake` only, so such a URL parsed and then failed at
connect with "TLS support not compiled in". The parser accepting a scheme the
client could not speak was the worst of both: it looked supported.

Enabled `rustls-tls-native-roots` -- the platform trust store rather than
bundled roots, so a private CA installed on the machine is trusted like a
public one.

**rustls then panicked.** Version 0.23 refuses to guess its cryptography
backend when crate features do not settle it, and the refusal is a panic at the
first connection, not an error. Dictating over `wss://` would have taken the
process down mid-sentence. `install_crypto_provider` installs ring once, from
every path that opens a connection.

**Termination is Caddy's job, not ours.** Obtaining and renewing certificates
is the hard part of TLS, and it is solved. `docker/compose.tls.yaml` adds a
Caddy service that does it; syrinx keeps speaking plaintext on a compose
network that never leaves the host.

That overlay is a separate file rather than a compose profile because compose
interpolates every variable in a file whether or not its profile is active: a
required `SYRINX_DOMAIN` in the main file broke the LAN deployment that has no
domain.

### Verified

Against a private CA and a real Caddy in front of the running container:
dictation over `wss://` works (15x real time), an untrusted issuer is refused
with `UnknownIssuer`, and a hostname the certificate does not cover is refused.
A TLS client that accepted anything would be worse than no TLS, because it
would look safe.
