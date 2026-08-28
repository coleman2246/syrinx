# Asking where the transcript goes

**Date:** 2026-08-27
**Status:** Approved design, pre-implementation

## Problem

"When I press the stream button in the GUI it should ask me the first time
where to save to."

It already does — once. `crates/syrinx-gui/src/main.rs:1135-1168` is a toggle:

```rust
let streaming = self.state.stream_to.is_some();
let label = if streaming { "Streaming" } else { "Stream…" };
...
if ui.button(label).on_hover_text(hover).clicked() {
    if streaming {
        action = Some(Request::SetStreamFile { path: None });
        self.status_line = Some("Stopped streaming to file".into());
    } else if let Some(p) = rfd::FileDialog::new()
        .set_file_name(save::filename_for(&save::timestamp()))
        .set_directory(save::default_dir())
        .save_file()
    { ... }
}
```

The dialog is reached only when `stream_to` is `None`. But the daemon persists
the chosen path into `~/.config/syrinx/config.toml`
(`crates/syrinx-client/src/daemon.rs:270-280`), so every subsequent run of the
GUI starts with `stream_to` already set. The button reads **Streaming** before
the user has done anything, and pressing it *stops* streaming instead of asking
where to write.

The consequence is worse than a confusing label. `save::timestamp()` is only
ever used to seed the dialog's default filename (`save.rs:393-404`); once
accepted, the literal string is stored. A file created on the 20th accumulates
every session thereafter, still named `2026-08-20_09-14-03.txt`. The user
believes they are streaming to today's transcript and are appending to a
fortnight of them.

Two smaller dishonesties sit alongside it:

- **The tooltip claims the present tense.** It says `Appending to {p}` as soon
  as the setting exists, but `session.rs:425-433` opens the `StreamWriter` once
  at session start, from `SessionOptions::stream` captured in
  `DaemonRuntime::start` (`daemon.rs:710-714`). Pressing Stream during a
  running session changes nothing about that session. The claim is about the
  next one.
- **Failures are shown as successes.** `App::send` (`main.rs:571-582`) routes
  a daemon `Response::Error` and a transport failure into `status_line`, which
  is rendered unconditionally in `theme::palette::SUCCESS` green
  (`main.rs:625-639`). A stream path that cannot be opened fails at
  `StreamWriter::open` inside `session::start`; a mid-session append failure
  only reaches `error!` in the daemon log and never the window at all.

## Design

### The button always asks

`Stream…` keeps its ellipsis — the GUI's convention is that `…` means a dialog
opens (`Stream…`, `Save as…`, `File…`) against bare verbs that act immediately
(`Save`, `Copy`, `Clear`, `Rescan`) — and it now honours it unconditionally.
Every press opens the Save dialog, seeded with:

- a **freshly timestamped filename**, `save::filename_for(&save::timestamp())`,
  so the default is always today's;
- the **remembered directory**: the parent of the persisted `stream_to`, falling
  back to `save::default_dir()` when there is none or its parent no longer
  exists.

The remembered path becomes a default rather than a commitment. Cancelling the
dialog leaves the current state exactly as it was — cancel is not a stop.

### Stopping is its own control

Because the button is no longer a toggle, a separate `Stop streaming` button is
rendered beside it, only while streaming. It sends the
`Request::SetStreamFile { path: None }` that the second press used to send. Two
plain buttons in a row matches the existing idiom in `App::controls`, which is
a single `ui.horizontal` holding every action and accumulating one
`Option<Request>` dispatched at the end (`main.rs:1102-1258`).

### The tooltip says which session it means

While a session is running, the hover text says the target takes effect on the
**next** session, because that is what the code does. While idle it says what
will be appended to. The existing `SourceMode::Separate` branch is kept: with
more than one source the text names one file per source beside the chosen path,
since `save::path_for_source` (`save.rs:331-343`) means the chosen path itself
may have nothing written to it.

The resolved target also moves out of the tooltip and into the window as a weak
label beneath the controls, so it is visible without hovering — including the
per-source split names when Separate mode is active.

### Errors look like errors

`status_line` gains a severity. A `Response::Error` or a transport failure
renders in `theme::palette::DANGER`; a success stays green. This is a small
change with an outsized effect on this feature specifically, since the most
likely failure of "pick a file" is picking one that cannot be opened, and today
that failure is reported in the colour of success.

Mid-session append failures (`session.rs:460-464`) additionally set
`SessionState::error`, so a stream that dies an hour into a meeting is visible
in the window rather than only in the daemon log.

## What is deliberately not changed

- **`stream_to` stays persisted.** Remembering the directory across runs is
  useful; it is the silent reuse of a stale *filename* that was wrong.
- **`StreamWriter` semantics.** Still `create(true).append(true)`, still opened
  once at session start so an unwritable path is refused before an hour of
  talking rather than after, still no rotation.
- **The separate-mode naming rule.** `save::path_for_source` and the
  `sources > 1` condition are correct and tested; this design only makes their
  output visible.

## Testing

There is no test today covering `Request::SetStreamFile` end to end through the
daemon handler, and none covering the GUI's Stream button at all — its 16 tests
are `normalise_server`, `tail`, `daemon_beside`, `no_window_advice`,
`daemon_log_path` and `window_icon`.

- **The dialog seed is recomputed per press**: a pure helper returning
  `(directory, filename)` from the current `stream_to` and clock, asserted to
  return today's stamp and the remembered directory, and to fall back to
  `save::default_dir()` when the remembered parent is gone.
- **Cancel does not stop streaming** — no `Request` is emitted.
- **`Stop streaming` sends `SetStreamFile { path: None }`**, and the daemon
  handler removes the key from the config file (the existing
  `clearing_an_optional_setting_removes_it` covers the config half).
- **`SetStreamFile` round-trips through the daemon** and is reflected in the
  published `DaemonState::stream_to`.
- **Tooltip and label text** differ between running and idle, asserted through
  the pure text helpers rather than through egui.
- **Severity**: an error response yields `DANGER`, a save yields `SUCCESS`.
