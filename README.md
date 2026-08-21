# syrinx

Self-hosted streaming speech-to-text, built on NVIDIA Parakeet/Nemotron
cache-aware streaming ASR via [`parakeet-rs`](https://github.com/altunenes/parakeet-rs).

Unlike whisper (which must see a complete utterance before emitting anything),
a transducer emits tokens as audio arrives. That gives word-level latency without
Vosk's accuracy penalty: ~1.93% WER on librispeech test-clean, with punctuation.

## Layout

| Crate | Purpose |
|---|---|
| `syrinx-proto` | Wire types. One definition of the protocol, shared by everything. |
| `syrinx-server` | WebSocket + HTTP service. Ships as a CUDA 12.8 container. |
| `syrinx-gui` | egui + cpal. Desktop and laptop, mic or system audio. |
| `syrinx` | Headless Linux dictation client, types at the cursor via `wtype`. |

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

That GPU is also shared with **Frigate** (live camera recording) and **Jellyfin**
(NVENC transcoding), both of which fail visibly when starved. The server is
therefore designed as the lowest-priority GPU tenant: it holds zero VRAM when
idle, checks free VRAM before loading, caps its own ORT arena, and refuses
sessions rather than winning an allocation race against the cameras or a film.

See the design document for measured model sizes and the full tenancy policy.

## Sway

The level overlay shown during typing modes is an ordinary Wayland window, so a
tiling compositor tiles it unless told otherwise. One rule makes it behave as an
overlay:

```
for_window [app_id="syrinx-overlay"] floating enable, sticky enable, border none, resize set 260 54
```

`sticky` keeps it visible across workspaces; without it the readout disappears
the moment you switch away from where dictation started, which is precisely when
you want it.

A keybind for dictation:

```
bindsym $mod+n exec syrinx toggle --mode type
```

## Typing into Electron apps (Teams, Discord, VS Code)

The default typing method is `wtype`, which uses the Wayland virtual-keyboard
protocol. It is correct for most native Wayland applications but **fails in
Electron and Chromium apps**: each call creates and destroys a virtual keyboard,
Chromium re-evaluates focus whenever input devices appear, and the text field
loses focus — so the keystrokes are interpreted as global shortcuts instead. In
Teams that looks like the chat list jumping about rather than a message being
typed.

`ydotool` writes to `/dev/uinput` instead. The kernel presents one persistent
virtual device that is indistinguishable from a real keyboard, so applications
that mishandle the virtual-keyboard protocol still receive it normally.

```bash
sudo pacman -S ydotool
systemctl --user enable --now ydotool     # the daemon; ydotool cannot type without it
```

Then in `~/.config/syrinx/config.toml`:

```toml
inject = "ydotool"
```

Options are `wtype` (default), `ydotool`, and `paste`. Paste copies to the
clipboard and sends Ctrl+V, restoring the previous clipboard afterwards — the
most broadly compatible of the three, though terminals need Ctrl+Shift+V so it
is a poor fit there.
