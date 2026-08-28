# Two sources on Windows, and the silence that stops both

**Date:** 2026-08-27
**Status:** Implemented. Revised after review; the design below is what was
built, including the places where the first draft of it was wrong.

## Problem

On Windows, selecting two sources — a microphone and system audio — sometimes
produces no input at all. Not degraded audio: none. The status line reads
`Listening`, no error appears anywhere, and the only recovery the user has
found is restarting the daemon and the GUI several times until it happens to
work.

That report is five separate faults wearing one symptom, and fixing only the
loudest of them would leave the bug half-cured. Fault 5 was found during
review, after the first four had been fixed; it reproduces the symptom on its
own, word for word.

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

### Fault 5: a panicking cpal call wedges the daemon at Listening

cpal's Windows backend panics rather than returning errors in several places,
and two of them — `find_device` and `default_input_config` /
`default_output_config` — run on the thread the daemon spawns to hold a
session. That thread dying takes the session's only chance to record what
happened with it: `SessionState.status` stays wherever it had reached and
`error` stays `None`.

Since the handshake sets `Status::Listening` before any device is opened, and
`Status::is_active()` is `!matches!(self, Idle)`, the dead session reads as
running for ever. `reap_finished_session` never fires, `stop()` sends into a
oneshot nobody holds, metering stays off. Listening, no error, no audio, Start
and Stop inert, recoverable only by restarting the daemon — the report, exactly.

The full list of reachable panic sites, and what is done about them, is under
"A panicking cpal call cannot take the process — or the session — down" below.

## Design

### A timeline mixer, not a rendezvous

Emission stops being driven by what every source has in common and starts being
driven by the clock. On each 20 ms tick the mixer emits exactly `FRAME = 320`
samples (20 ms at 16 kHz). Each source contributes the samples it has; a source
short of a full frame contributes silence for the remainder.

This is the whole fix for Fault 1. A source that produces nothing contributes
silence and the mix continues, which is both the correct rendering of an idle
loopback and the behaviour a user expects.

Gain is `1/N` over the sources that are **live**, where live means "not
starved" by the same `STARVE_AFTER` test the health rows use.

Dividing by every selected source instead was the first draft of this, on the
reasoning that `mix_frames` already averaged across all of them so nothing was
being changed. That reasoning does not hold. It is true only while both
sources are live, and the case this design exists for — a microphone beside a
Windows loopback with nothing playing — is exactly the case where they are
not: before the clock, the mix emitted *nothing at all* there, so there was no
prior behaviour to preserve. Dividing by two would have put the microphone on
the wire at half amplitude for the whole session, permanently, in the
commonest two-source configuration there is.

The hazard that reasoning was guarding against is real but narrower than it
looked: a divisor that followed what arrived in the last 20 ms would move the
output 6 dB every time a speaker paused between words. Starvation is five
seconds of hysteresis. No utterance is long enough to straddle it, so the
divisor can only change on an edge that is already slow — which is what makes
it safe to use.

Per-source queues are bounded at `MAX_QUEUED = 3_200` samples (200 ms) rather
than 2 s. The old bound existed to absorb the rendezvous; with the clock
driving emission, a queue that keeps growing means that source is running fast
relative to the tick — which happens for real, because `resample_to_16k` uses
`chunks_exact(factor)` (`crates/syrinx-proto/src/audio.rs:59-62`) and silently
drops up to `factor - 1` samples per callback whenever the WASAPI period frame
count is not a multiple of the ratio. Trimming at 200 ms keeps the sources
loosely aligned instead of letting one drift two seconds ahead. Trims are
logged, rate-limited to at most one message per source per five seconds, and
the message reports the **cumulative** count: trimming happens on nearly every
callback once it starts, so the figure from the one call that won the rate
limit understates the loss by orders of magnitude. It also does not blame the
source, because a queue overflows when the source outruns the clock and
equally when the mix falls behind its own tick, and the queue cannot tell
which. The cumulative count travels to the window on the health row too — words
going missing mid-utterance is not something a reader can otherwise discover.

The tick uses `MissedTickBehavior::Burst`. `Delay` looks like the safe choice
and is not: tokio applies missed-tick behaviour whenever a tick runs more than
5 ms late, which under load is routine, and `Delay` restarts the schedule from
the moment it noticed. The frames that fell due while it was late are never
taken — and they are not skipped either, because they are sitting in the
source's queue. They are held there and then trimmed away as overflow.
Measured under a consumer stalling 60 ms once a second: `Delay` discarded
180 ms of audio per source per ten seconds and pinned both queues at their
bound, adding 200 ms of latency permanently; `Burst` dropped nothing and left
the queues empty.

Emission also has to happen *before* a source is judged spent, not after.
`take_frame` is what empties a queue, so a spent test taken afterwards is true
on the very tick that produced the final frame — and that frame was then
dropped. Live this lost the last 20 ms of every source at every stop, and lost
everything when a capture ended with less than one frame buffered.

### One source that fails to open costs only itself

`MixedCapture::start` skips a source whose device will not open and starts the
rest, matching what `DaemonRuntime::start` already does with a source key that
no longer resolves. Failing the whole session for it would take away the
working microphone beside the broken loopback, which is the opposite of what
selecting two sources is for. The skipped source keeps its place in the row
order and carries its reason on that row, so the window can say which source is
missing and why rather than showing one row where two were ticked. Every source
failing is still an error: there is nothing left to record, and an empty mix
would report `Listening` for ever.

### Silence that says so

The mixer tracks, per source, the total real samples received and the instant
the last non-empty contribution arrived. A source that has contributed nothing
for `STARVE_AFTER = 5s` is reported as silent.

This travels to the window as a new per-source health row on `DaemonState`
(`crates/syrinx-client/src/ipc.rs`), carrying the source's `short_label()`, its
RMS, whether it is currently starved, how many samples have been trimmed from
it, and why it is contributing nothing when its device would not open. The GUI
renders one meter row per selected source instead of the single post-mixer
meter it shows today.

Per **selected** source, not per source that is reporting one. Only sources
that started report, so rows built from what is reporting show nothing at all
in the two-sources-one-broken case — which is the case the rows exist for, and
the one row that would be missing is the broken one the user is looking for. A
selected source with no row of its own gets one saying so.

The same holds for the levels while nothing is running. A window that falls
back to the last finished session's frozen RMS and silent flags is presenting
stale data as live, which is worse than presenting none: the two cannot be
told apart. The transcript survives a stop, because that is there to be read
and saved; the meters do not, because a meter is a claim about right now.

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

Device lookup and the config query move onto that same thread. They are the
two cpal calls that used to run on the caller's, and both panic on Windows
rather than returning (see below). The whole closure runs inside the panic
guard, so a panic anywhere in it reaches the caller as the failed open it is,
carrying the panic's own message.

The rendezvous means `Capture::start` can now block for as long as the open
takes, where before it returned at once. Anything holding a loop must
therefore not call it inline — see the preview below.

### The preview lets go, and does not hold anything up

The preview task selects over `rx.recv()` and its stop signal, so a source that
never delivers a chunk can no longer pin the capture. The stop flag becomes a
`tokio::sync::Notify` (or a watch channel) rather than a bool checked after a
blocking await, and `_capture` is dropped explicitly on exit.

`Preview::start` returns immediately and reports what became of the device
through a state its holder polls. It cannot wait, because the daemon starts
these inline in the loop that dispatches IPC, updates the tray and publishes
state — and that loop's clients give every request five seconds before
reporting "daemon did not answer in time". Waiting there for a ten-second open,
once per selected source, makes Start, Stop and the source picker fail and
freezes the tray.

It also returns a `Preview` whatever happens, rather than an `Err` on the paths
that fail. An error return before anything owns the stop signal leaves a thread
still inside `Capture::start`, which goes on to acquire a capture nobody can
release — the same leak by a different route. The caller always holds the
handle that stops it.

A selection that produces no meter at all is retried on a timer rather than on
the next tick. Retrying at 40 Hz reopens a failing device continuously, holding
the loop for as long as each attempt takes to fail, and leaks on every pass if
anything about the failure path leaks.

Previews are released at the top of `DaemonRuntime::start`, before the session
asks for anything. Ordering alone was enough in practice — `Status::Connecting`
is set synchronously, so the next `update_preview` in the same tick would drop
them, and the session does its WebSocket handshake before touching a device —
but that is an accident of ordering rather than a promise, and per-source
metering took the exposed surface from one endpoint to one per source.

### The selection survives

`Config` gains `source_keys: Vec<String>` and `source_mode: SourceMode`,
persisted by `Request::SetSources` exactly as `SetFormat` persists `format`.
The existing singular `source_key` is still read as a fallback when
`source_keys` is absent, so an existing config keeps working, and is written
through as the first element so a downgrade keeps working too.

Everything that reads a selection reads it through `Config::selected_sources`
— the daemon, `syrinx run` and `syrinx meter` alike. Leaving the last two on
the singular key would make them work only by accident, on the mirroring
above, and would silently ignore a config written by hand with `source_keys`
and no `source_key`. `syrinx run` reads `source_mode` for the same reason:
without it, a selection the window saved as separate comes back combined and
mixes two people into one unlabelled stream.

Settings are written back to the config file the daemon was **given**, not to
the canonical path. A daemon started as `syrinx daemon --config <other>` that
saved elsewhere would drop the selection on its next restart, which is the
fault this section exists to fix.

### A panicking cpal call cannot take the process — or the session — down

cpal's Windows backend does not confine itself to returning errors. It panics
on a transient failure at an endpoint in at least four places this code
reaches:

- `Device::description()` → `.OpenPropertyStore(STGM_READ).expect("could not
  open property store")`
- `default_input_config()` / `default_output_config()` → `data_flow()` →
  `.expect("could not query IMMDevice interface for IMMEndpoint")` and
  `.expect("could not get endpoint data_flow")` (`host/wasapi/device.rs:179`,
  `:185`)
- `input_devices()` / `output_devices()` → `get_enumerator()` →
  `CoCreateInstance(..).unwrap()` (`device.rs:1177`)
- `Devices::next()` → `self.collection.Item(i).unwrap()` (`device.rs:1292`),
  which fails when an endpoint disappears between the collection being
  snapshotted and the entry being read — the device-swap case itself

Those are reached from the GUI's UI thread every two seconds, from the daemon's
main loop, and — for the lookup and the config query — from the thread that
starts a session.

The session thread is the worst of the three, and is the reported symptom
exactly. The panic kills that thread before anything records a failure, so
`SessionState.status` stays wherever it had reached and `error` stays `None`.
Since `Status::is_active()` is `!matches!(self, Idle)`, the daemon reads the
dead session as running for ever: it is never reaped, `stop()` sends into a
oneshot nobody holds, and metering stays off. Listening, no error, no audio,
Start and Stop inert, recoverable only by restarting the daemon.

Three things follow.

`list_sources()` and `find_device()` guard the enumeration itself as well as
the per-device description, and a skipped endpoint is logged once. A panic
while reading an entry ends the listing rather than skipping past it, because
cpal increments its index only *after* the unwrap — asking again would panic on
the same entry for ever.

The two calls made on the session thread move onto the capture's own stream
thread, inside the same guard, so what the session sees is a failed open.

And the session thread catches the unwind at its own boundary, whatever the
cause, routing it through the existing `fail()` path. That is the part that
matters: it makes the whole class survivable rather than this one instance of
it. `Status::Listening` also moves to after the captures open, so it means
listening rather than intending to.

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
- **A starved source is left out of the gain divisor**, so the live one is not
  attenuated by it. The starvation clock is reached into rather than waited
  out; five seconds of sleeping on every run is not a test.
- **The mix ends only after emitting what its sources left behind**: one frame
  pushed, both senders dropped, and that frame has to come out.
- **A mix that ran late catches up** rather than emitting on a schedule that
  restarted, and trims nothing while doing it.
- **A source that would not open neither silences nor hides itself**: the
  working source is unattenuated, and the failed one keeps its row and its
  reason. **Every source failing is an error**, not an empty stream.
- **`Capture::start` returns the build error** instead of `Ok` — exercised
  through a seam that lets the test force a build failure. **A backend that
  panics** reaches the caller as an error carrying the panic's own message.
- **A session thread that panics ends failed rather than Listening for ever**,
  with the reason where a window can show it, and neither a clean end nor an
  ordinary error is disturbed by the guard.
- **Starting a preview does not wait for its device**, and a device that will
  not open says so rather than reading zero.
- **Meters that all failed are retried on a timer**, while a new selection
  re-points them at once.
- **Two sources with one broken still get two rows** in the window, and a
  source with no row of its own is named from what was enumerated.
- **A finished session stops reporting its last levels as live**, while its
  transcript stays.
- **Every path reads the selection the same way**: a config holding only
  `source_keys` is honoured, every remembered source is used rather than only
  the first, and the remembered `source_mode` is read when no flag overrides
  it.
- **A daemon given a config writes its selection back to that file.**
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
