# parakeet-stt

Self-hosted streaming speech-to-text, built on NVIDIA Parakeet/Nemotron
cache-aware streaming ASR via [`parakeet-rs`](https://github.com/altunenes/parakeet-rs).

Unlike whisper (which must see a complete utterance before emitting anything),
a transducer emits tokens as audio arrives. That gives word-level latency without
Vosk's accuracy penalty: ~1.93% WER on librispeech test-clean, with punctuation.

## Layout

| Crate | Purpose |
|---|---|
| `parakeet-proto` | Wire types. One definition of the protocol, shared by everything. |
| `parakeet-server` | WebSocket + HTTP service. Ships as a CUDA 12.8 container. |
| `parakeet-gui` | egui + cpal. Desktop and laptop, mic or system audio. |
| `parakeet-type` | Headless Linux dictation client, types at the cursor via `wtype`. |

## Modes

- **live** — near-real-time, typed at the cursor. Append-only: the server never
  asks the client to delete text, because it is typing into arbitrary
  applications where that would be destructive.
- **transcript** — running transcript in a GUI the user can save. The client owns
  the buffer, so the server may revise it.
- **bulk** — offline file transcription over HTTP, using the more accurate
  non-streaming model.

## Status

Design approved, implementation not started.
See [`docs/specs/`](docs/specs/) for the design document.

## Deployment notes

The target server (`acdc`, Ubuntu, RTX 2070 Super) runs driver 570, which caps
CUDA at **12.8** — a CUDA 13 image will not run there. The container pins CUDA
12.8, which also works on the CUDA 13.3 development desktop by backward
compatibility. One image, both machines.

That GPU is shared with Frigate, a live camera recorder. The server must never
exhaust VRAM. See the design document for the budget and the mitigations.
