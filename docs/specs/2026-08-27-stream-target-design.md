# Asking where the transcript goes

**Date:** 2026-08-27
**Status:** Implemented, and amended after review

The sections marked *added after review* are the parts this document did not
get right the first time. The largest is under "The name is refreshed wherever
a session starts": everything below was written about the GUI's Stream button,
and the complaint that started the document is about a file that keeps its
name — which nothing about a dialog can fix for the tray, the hotkey, `Ctrl+D`
or `syrinx start`, none of which open one.

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

### The name is refreshed wherever a session starts

*Added after review: the design above fixes only the presses that go through
the dialog, and most sessions do not.* The tray, the global hotkey, `Ctrl+D`,
`syrinx toggle` and `syrinx start` all reach `DaemonRuntime::start`, which
reads `config.stream_path()` — the persisted path, filename and all. Seeding
a dialog they never open changes nothing for them, so the complaint that
started this document survives on every path but one.

So the decision belongs beside the names it is about, in `save.rs`:

```rust
pub fn restamped(path: &Path, stamp: &str) -> PathBuf
```

An **auto-generated** name is re-stamped; a **deliberate** name is respected.
A file name whose stem has the shape `save::timestamp()` produces —
`%Y-%m-%d_%H-%M-%S`, any extension — and whose date is not today's is replaced
with a freshly stamped one in the same directory, keeping its extension.
Anything else is returned untouched: `notes.txt` was chosen on purpose, and
"stopping and starting continues where you left off" has to keep meaning that
for whoever chose it.

`DaemonRuntime::start` calls it before building `SessionOptions::stream`, and
**writes the result back** through `set_stream_file`. Written back rather than
merely used, for two reasons: a fresh stamp minted per session would give every
session of the day its own file, and every viewer's label reads the persisted
setting, so a value that was not written back would name a file that is not the
one being written. It also runs once at daemon startup, so a window opened the
morning after does not show yesterday's name until something starts.

### Clearing the file is its own control

Because the button is no longer a toggle, a second button is rendered beside
it, only while a file is set. It sends the
`Request::SetStreamFile { path: None }` that the second press used to send. Two
plain buttons in a row matches the existing idiom in `App::controls`, which is
a single `ui.horizontal` holding every action and accumulating one
`Option<Request>` dispatched at the end (`main.rs:1102-1258`).

*Added after review:* it is labelled **Clear stream file**, not `Stop
streaming`. `session::run` opens the writer once, at session start, so nothing
sent now reaches a session already writing — a control that said "stop" while
one was running would be telling the user the recording had stopped when it
had not. And while nothing is running there is no activity to stop either:
what the button does, in both states, is clear a setting. Its hover text
distinguishes the two the same way the `Stream…` hover does.

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

*Added after review:* the label is running-aware too, and for the same reason
the tooltip is. A target changed mid-session would otherwise leave the window
reading `Stream target: B.txt` while every fragment went to `A.txt`, and the
label is the more visible surface of the two — being visible without hovering
is why it exists.

It also names the split files only when it can. `DaemonRuntime::start` resolves
each remembered key against the sources present, **skips any that have gone**,
and splits between what is left; the window resolves against its own scan,
which can be older. With one of two devices unplugged, the window named two
files and the daemon opened one, so none of the names shown was ever written —
and an unresolved key was turned into a filename of its own, inventing
`notes-alsa-input-usb-blue-microphones-yeti-st.txt` from a config string. The
label claims a per-source split only when every selected key resolves, and
otherwise names the chosen path.

### Errors look like errors

`status_line` gains a severity. A `Response::Error` or a transport failure
renders in `theme::palette::DANGER`; a success stays green. This is a small
change with an outsized effect on this feature specifically, since the most
likely failure of "pick a file" is picking one that cannot be opened, and today
that failure is reported in the colour of success.

Mid-session append failures (`session.rs:460-464`) reach the window rather than
only the daemon log, so a stream that stops taking words an hour into a meeting
does not look like one still being written.

*Added after review:* not in `SessionState::error`, which is what the CLI exits
non-zero on and what the window paints red. A failed append costs the fragment
it was carrying and nothing else — the writer stays open, the next fragment may
well succeed, the transcript in memory is untouched — so `syrinx start --stream`
exited non-zero after a complete and fully saved transcript because a USB stick
blinked once, and in separate mode a blip on one source took the one `error`
slot `merge_states` keeps and hid another source that had really died. It gets
a field of its own, folded separately, worded as what is known (a fragment was
not written) rather than as a stream that stopped, and painted amber.

## What is deliberately not changed

- **`stream_to` stays persisted.** Remembering the folder across runs is
  useful, and so is remembering a name the user chose; it is the silent reuse
  of a stale *generated* filename that was wrong.
- **`StreamWriter` semantics.** Still `create(true).append(true)`, still opened
  once at session start so an unwritable path is refused before an hour of
  talking rather than after, still no rotation.
- **The separate-mode naming rule.** `save::path_for_source` and the
  `sources > 1` condition are correct and tested; this design only makes their
  output visible.

  *Added after review:* one thing feeding that rule was not correct.
  `Source::short_label` calls every monitor "System audio", so two monitors in
  separate mode built one filename and pointed two `StreamWriter`s at it —
  precisely the tearing file-per-source exists to prevent. Names are now
  produced for a whole set at once by `syrinx_audio::source::short_labels`,
  which suffixes a collision.

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
- **`Clear stream file` sends `SetStreamFile { path: None }`**, and the daemon
  handler removes the key from the config file (the existing
  `clearing_an_optional_setting_removes_it` covers the config half).
- **`SetStreamFile` round-trips through the daemon** and is reflected in the
  published `DaemonState::stream_to`.
- **Tooltip and label text** differ between running and idle, asserted through
  the pure text helpers rather than through egui.
- **Severity**: an error response yields `DANGER`, a save yields `SUCCESS`.

*Added after review:*

- **`save::restamped`**: a generated name from an earlier day gets today's
  date; one from today is left where it is, so two sessions in a day meet in
  one file; a name the user chose is never touched; an unusual extension
  survives; a path with no folder in front of it and one that names no file at
  all behave sanely.
- **`DaemonRuntime::refresh_stream_name`**: the refreshed name is applied *and*
  persisted, a second call the same day changes nothing, and a name the user
  chose does not even cause a config write.
- **A lost fragment** is reported without setting `error`, is not erased by a
  later success, is not replaced by a second failure, and does not hide a real
  error in `merge_states`.
- **Colliding source names**: two monitors never resolve to one name, and a
  lone monitor keeps the name it always had.
- **The window fits in the window**: two tests drive a real `egui` context at
  the window's size and measure what comes out — nothing wider than the
  window, the server address still in the corner, nothing painted below the
  bottom edge. These are the only tests here that touch egui, and they are
  worth it: what they check is exactly what cannot be seen by reading the
  code.
