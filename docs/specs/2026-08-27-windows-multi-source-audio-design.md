# Two sources on Windows, and the silence that stops both

**Date:** 2026-08-27
**Status:** Approved design, pre-implementation

## Problem

On Windows, selecting two sources — a microphone and system audio — sometimes
produces no input at all. Not degraded audio: none. The status line reads
`Listening`, no error appears anywhere, and the only recovery the user has
found is restarting the daemon and the GUI several times until it happens to
work.

That report is three separate faults wearing one symptom, and fixing only the
loudest of them would leave the bug half-cured.

### Fault 1: the mixer waits for the quietest source, forever

`crates/syrinx-audio/src/mixer.rs:82-93` decides how much audio to emit by
taking the minimum across every source's queue:

```rust
let available = queues.iter().map(|q| q.lock()...len()).min().unwrap_or(0);
if available == 0 { continue; }
```

If any one source has produced nothing, `available` is zero, the `continue`
fires, and **nothing is emitted for any source**. This is not a deadlock —
nothing blocks — it is a starvation gate, which is why it presents as silence
rather than a hang. Downstream, `session.rs:529` never sees a chunk, so
`state.levels` and `state.rms` are never written and no error is ever set.

On Windows this is reached by ordinary use. WASAPI loopback on an idle render
endpoint delivers no packets whatsoever: cpal's input thread blocks in
`WaitForMultipleObjectsEx(..., INFINITE)` waiting on an event the audio engine
never sets, and `process_input` returns immediately on
`GetNextPacketSize() == Ok(0)`. A user who selects mic + system audio and does
not happen to have something playing has selected a source that produces
nothing, and has thereby silenced their microphone too.

Note the asymmetry that makes this a Combined-mode bug specifically: the mixer
is only reached when a single session holds more than one source
(`crates/syrinx-client/src/session.rs:401-411`). `SourceMode::Separate` gives
each session exactly one source and never mixes. Combined is the default the
GUI lands on when a second box is ticked.

Even when it does work, the gate distorts. While the loopback is silent the
microphone queue fills to `MAX_QUEUED = 32_000` samples (2.0 s) and trims from
the front (`mixer.rs:73-75`); when playback resumes, `available` is the
loopback's short length, so the microphone audio that finally gets mixed is up
to two seconds stale and has had content discarded in between.

### Fault 2: a capture that never started still reports success

`crates/syrinx-audio/src/capture.rs:88-136` spawns a thread and returns before
that thread has built or played anything:

```rust
std::thread::spawn(move || {
    let stream = match format { ... };
    let Ok(stream) = stream else {
        tracing::error!("failed to build the input stream");   // the error value is dropped
        return;
    };
    if stream.play().is_err() {
        tracing::error!("failed to start the input stream");   // and here
        return;
    }
    let _ = stop_rx.recv();
});
Ok(Self { _stop: stop_tx })
```

A build or play failure on either source therefore returns `Ok`, logs without
the error value, and leaves a live handle that produces nothing — which,
through Fault 1, silences the whole mix. `MixedCapture::start` returns `Ok`,
the session proceeds to its WebSocket handshake, the daemon reports
`Listening`, and `SessionState::error` stays `None`. `Capture::start` only
propagates two failures today: `find_device` and `default_*_config`
(`capture.rs:62-78`). Everything after those is invisible.

An unsupported sample format takes the same path: `capture.rs:90-122` logs and
returns from the thread, leaving the handle alive.

### Fault 3: a silent source pins a capture for the life of the daemon

`crates/syrinx-client/src/preview.rs` is the idle level meter. Its task loops
`while let Some(chunk) = rx.recv().await` and checks its stop flag only *after*
a chunk arrives (`preview.rs:68-71`). `Drop` merely sets that flag
(`preview.rs:109-113`).

The `tx` lives inside the cpal callback, which is kept alive by `_capture`,
which is kept alive by the task. So when the metered source never delivers a
chunk — a silent WASAPI loopback, exactly Fault 1's condition — `rx.recv()`
returns neither `Some` nor `None`, the task never exits, and the WASAPI capture
client, its thread and its runtime are held until the process dies.
`Preview::start` cannot detect this; it returns `Ok` as soon as
`Capture::start` returns, which is immediate.

The daemon re-points the preview whenever the first selected key changes
(`daemon.rs:533-556`), leaking another each time. This is the mechanism behind
"restart the daemon *and* the GUI a few times": handles accumulate across GUI
open/close cycles and only a fresh daemon process clears them.

### Fault 4: the two-source selection does not survive a restart

`Request::SetSources` stores into `opts.source_keys` / `opts.source_mode` and
**never persists** (`daemon.rs:245-257`), unlike `SetFormat`, `SetStreamFile`
and `set_diarize`, all of which call `Config::save`. The client config has only
a singular `source_key` (`crates/syrinx-client/src/config.rs:26-27`) and no key
for `SourceMode` at all.

So restarting the daemon silently resets to a single source, chosen from
`config.source_key` or by `choose_source(..., None)` — which never picks a
Monitor (`crates/syrinx-client/src/lib.rs:68-74`). The user restarts, gets one
working source, re-ticks the second box, and is back in Fault 1. That is the
shape of "a few times before it works".

## Design

### A timeline mixer, not a rendezvous

Emission stops being driven by what every source has in common and starts being
driven by the clock. On each 20 ms tick the mixer emits exactly `FRAME = 320`
samples (20 ms at 16 kHz). Each source contributes the samples it has; a source
short of a full frame contributes silence for the remainder.

This is the whole fix for Fault 1. A source that produces nothing contributes
silence and the mix continues, which is both the correct rendering of an idle
loopback and the behaviour a user expects.

Gain stays at `1/N` — `mix_frames` already averages across all sources
(`mixer.rs:25-34`), so this is what already happens whenever both sources are
live and no level change is being introduced. Deliberately *not* switching to
"average over sources that had data": that would make the output level jump by
6 dB the moment a second source started or stopped producing, in the middle of
an utterance, which is worse for the ASR than a constant halving.

Per-source queues are bounded at `MAX_QUEUED = 3_200` samples (200 ms) rather
than 2 s. The old bound existed to absorb the rendezvous; with the clock
driving emission, a queue that keeps growing means that source is running fast
relative to the tick — which happens for real, because `resample_to_16k` uses
`chunks_exact(factor)` (`crates/syrinx-proto/src/audio.rs:59-62`) and silently
drops up to `factor - 1` samples per callback whenever the WASAPI period frame
count is not a multiple of the ratio. Trimming at 200 ms keeps the sources
loosely aligned instead of letting one drift two seconds ahead. Trims are
logged, rate-limited to at most one message per source per five seconds.

### Silence that says so

The mixer tracks, per source, the total real samples received and the instant
the last non-empty contribution arrived. A source that has contributed nothing
for `STARVE_AFTER = 5s` is reported as silent.

This travels to the window as a new per-source health row on `DaemonState`
(`crates/syrinx-client/src/ipc.rs`), carrying the source's `short_label()`,
its RMS, and whether it is currently starved. The GUI renders one meter row per
selected source instead of the single post-mixer meter it shows today.

The wording is *silent*, not *failed*. An idle Windows loopback producing
nothing is correct behaviour, and a red error for it would be a lie. What the
user needs is to see **which** source is contributing, which today is
impossible: metering while a session runs is taken downstream of the mixer
(`session.rs:536-540`), and the GUI suppresses the numeric readout entirely
while running, printing `"(session running)"` (`crates/syrinx-gui/src/main.rs:905-913`).
A running session at 0% and one at 40% look identical.

The idle preview also meters only `source_keys.first()` (`daemon.rs:533`), so
the second source is never metered anywhere at any time. It gets a preview too.

### Captures report what actually happened

`Capture::start` gains a rendezvous: the spawned thread builds the stream,
plays it, and sends the outcome back over a `std::sync::mpsc` channel before
entering its wait; `start` blocks on that recv and returns the real error. The
stream stays on its own thread — it is not `Send` on every platform, which is
why the thread exists — but the caller now learns whether it is running.

Both error sites log the error value they currently discard. The unsupported
sample-format arm becomes a returned error rather than a logged early return.

### The preview lets go

The preview task selects over `rx.recv()` and its stop signal, so a source that
never delivers a chunk can no longer pin the capture. The stop flag becomes a
`tokio::sync::Notify` (or a watch channel) rather than a bool checked after a
blocking await, and `_capture` is dropped explicitly on exit.

### The selection survives

`Config` gains `source_keys: Vec<String>` and `source_mode: SourceMode`,
persisted by `Request::SetSources` exactly as `SetFormat` persists `format`.
The existing singular `source_key` is still read as a fallback when
`source_keys` is absent, so an existing config keeps working, and is written
through as the first element so a downgrade keeps working too.

### Enumeration cannot take the process down

cpal's `Device::description()` does
`.OpenPropertyStore(STGM_READ).expect("could not open property store")`. That
is reachable from the GUI's UI thread every two seconds
(`crates/syrinx-gui/src/main.rs:528-537`) and from the daemon's main loop
(`daemon.rs:190`, `:546`, `:674`), for every endpoint on the machine. A
transient failure on any one endpoint currently panics whichever process
touched it first.

`list_sources()` wraps the per-device description call in `catch_unwind`,
skipping an endpoint that panics rather than propagating. A skipped endpoint is
logged once.

## Testing

The existing mixer tests cover `mix_frames` arithmetic only
(`mixer.rs:113-162`); nothing tests `MixedCapture::start`, the gate, or a
source that produces nothing. That is the gap that let this ship.

- **A source that never produces does not silence the others.** The regression
  test for Fault 1: two synthetic sources, one of which never sends, asserts
  the mixed output contains the live source's audio. Deadline-bounded so a
  regression fails rather than hangs, following `e11ee88`'s precedent.
- **A source that stops mid-stream does not silence the others**, and resumes
  cleanly when it returns.
- **Emission is clock-paced**: N ticks produce N × 320 samples regardless of
  what arrived.
- **A fast source is trimmed, not allowed to drift**, and the trim is logged
  at most once per interval.
- **Starvation is reported** after the threshold and cleared when data returns.
- **`Capture::start` returns the build error** instead of `Ok` — exercised
  through a seam that lets the test force a build failure.
- **The preview task exits when its source never produces a chunk**, deadline-
  bounded.
- **Config round-trips `source_keys` and `source_mode`**, and a config
  carrying only the legacy `source_key` still resolves to one source.

Manual verification on Windows, since none of the above proves the WASAPI
behaviour: `syrinx meter --source <key>` on each source separately, then two
sources combined with nothing playing, confirming the microphone still reaches
the transcript.

## Out of scope

Sample-rate drift correction between two devices running on independent
clocks. The 200 ms bound keeps drift bounded and audible-quality effects are
negligible at ASR's tolerance; a real resampler with drift tracking is a much
larger piece of work and nothing observed requires it yet.
