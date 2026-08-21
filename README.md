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

The default is **type at the cursor**: dictation into whatever you are working
in is the point, and a transcript you then have to copy across is a step in the
way. Set `mode = "transcribe"` for a transcript instead, or `"both"`.

- **live** — near-real-time, typed at the cursor. Append-only: the server never
  asks the client to delete text, because it is typing into arbitrary
  applications where that would be destructive.
- **transcript** — running transcript in a GUI the user can save. The client owns
  the buffer, so the server may revise it.
- **bulk** — offline file transcription over HTTP, using the more accurate
  non-streaming model.

## Status

Working on Linux and Windows. The server runs on the development desktop; the
container that moves it to the GPU host is the last piece outstanding.

See [`docs/specs/`](docs/specs/) for the design document.

## Deploying

See [docs/deploying.md](docs/deploying.md) for a GPU server behind an existing
nginx. The short version is below.

## Running the server in a container

```bash
docker build -f docker/Dockerfile -t syrinx-server:latest .

cp docker/.env.example docker/.env      # then fill in the token and model path
docker compose -f docker/compose.yaml up -d
```

`docker/.env` is gitignored and read automatically, so `ps`, `logs` and `down`
work without exporting anything:

```bash
docker compose -f docker/compose.yaml ps      # shows (healthy)
docker compose -f docker/compose.yaml logs -f
docker compose -f docker/compose.yaml down
```

The token comes from the environment: baking one into a layer publishes it to
anyone who can pull the image, and committing one publishes it to anyone who
can read this repository. Models are volume-mounted read-only — they are 2.5 GB
and change on a different schedule from the code.

The container runs unprivileged with a read-only root filesystem, all
capabilities dropped, and `no-new-privileges`.

The port is published on every interface, because other machines on the LAN are
the reason the service exists; access is gated on the shared token, which fails
closed when unset. Narrow it with `SYRINX_LISTEN=127.0.0.1:8770` on a machine
that only ever talks to itself.

### Reaching it from outside the LAN

The LAN default is plaintext `ws://`, which is fine on a network you control
and **not** something to forward a port to. There is no encryption there: the
bearer token travels in the clear on every connection, and so does the audio
and every transcript coming back. Anyone on the path reads the token once and
has permanent access to a microphone service.

You do not visit a certificate authority to fix this. Let's Encrypt *is* one,
it is free, and both options below talk to it for you — request, prove control
of the domain, install, renew.

**If a reverse proxy already terminates TLS on the host** (nginx, Traefik), add
a server block for syrinx rather than running a second one: see
`docker/nginx-syrinx.conf.example`. Get the certificate with the setup you
already have, e.g.

```bash
sudo certbot certonly --webroot -w /var/www/certbot -d dictate.example.com
```

The only part that needs care is the WebSocket upgrade. nginx will not proxy a
WebSocket without `proxy_http_version 1.1` and explicit `Upgrade`/`Connection`
headers; without them the handshake gets a plain 200 back and the client
reports a protocol error that looks like a syrinx bug.

**If nothing is on 80 and 443 yet**, the Caddy overlay is less work:

```bash
# in docker/.env
SYRINX_DOMAIN=dictate.example.com     # a dynamic-DNS name is fine
SYRINX_ACME_EMAIL=you@example.com     # optional; renewal warnings go here

docker compose -f docker/compose.yaml -f docker/compose.tls.yaml up -d
```

Caddy obtains and renews a Let's Encrypt certificate by itself — that is the
part of "just use TLS" that is actually hard. Forward ports **80 and 443** to
the host: 80 is not optional, it is how the certificate is issued and renewed.

Then point the client at it:

```toml
server = "wss://dictate.example.com"
```

No port needed; `wss://` means 443. The client validates the certificate chain
against the platform trust store and refuses on a bad one — verified against a
private CA: an untrusted issuer fails with `UnknownIssuer` rather than
connecting anyway.

Two things worth doing at the same time:

- **Generate a real token.** `openssl rand -hex 24`. On the internet the token
  is the only thing between a stranger and a transcription service running on
  your GPU. `dev-token` is not that.
- **Keep the LAN port closed at the router.** Publishing 8770 on the host is
  for machines already on the LAN; the internet should only ever reach 443.

## Deployment notes

The target server (Ubuntu, RTX 2070 Super) runs driver 570, which caps
CUDA at **12.8** — a CUDA 13 image will not run there. The container pins CUDA
12.8, which also works on the CUDA 13.3 development desktop by backward
compatibility. One image, both machines.

That GPU is also shared with **Frigate** (live camera recording) and **Jellyfin**
(NVENC transcoding), both of which fail visibly when starved. The server is
therefore designed as the lowest-priority GPU tenant: it holds zero VRAM when
idle, checks free VRAM before loading, caps its own ORT arena, and refuses
sessions rather than winning an allocation race against the cameras or a film.

See the design document for measured model sizes and the full tenancy policy.

## Windows

Both the CLI and the GUI build and run on Windows with the MSVC toolchain.
Verified on Windows 11 with a client on the LAN and the server on the desktop:
system audio captured, streamed, and transcribed live.

What differs from Linux:

| | Linux | Windows |
|---|---|---|
| Config | `~/.config/syrinx/` | `%APPDATA%\syrinx\` |
| Daemon IPC | Unix socket in `$XDG_RUNTIME_DIR` | named pipe `\\.\pipe\syrinx.sock` |
| Typing (`auto` resolves to) | `ydotool` if its daemon runs, else `wtype` | `sendinput` |
| Stopping a session | SIGTERM | a stop-request file, polled |
| Audio | PipeWire, including per-application capture | WASAPI via cpal |
| System tray | ksni (StatusNotifierItem) | tray-icon |
| Global hotkey | compositor binding | built in |

## One config, two machines

The generated config is **byte-identical on every platform**: same settings,
same defaults, same comments. It once listed only what the generating machine
could do, which read well until the same file was kept for a desktop and a
laptop — it was then wrong on whichever machine had not written it.

`inject = "auto"` is what makes the *default* portable rather than just the
file. It resolves per machine: `sendinput` on Windows; on Linux `ydotool` when
its daemon is running, otherwise `wtype`. Running `ydotoold` is a deliberate act
— a daemon to install, enable, and grant `/dev/uinput` — so it is taken as
consent, which means `auto` is also right in Electron apps where `wtype` fails.

Anything set explicitly is honoured exactly as written, and refused with a clear
message if it cannot work on that platform.

## Hotkey

`hotkey = "ctrl+alt+d"` in the config starts and stops dictation from anywhere.
Modifiers are `ctrl`, `alt`, `shift` and `super`; function keys work on their
own. Unset by default, because claiming a combination for the whole desktop is
not something to do to someone unasked.

**It only registers on Windows and X11.** Wayland has no way for a client to
grab a key globally — the compositor owns input, and a client that could do this
could keylog every other client. On Sway, GNOME or KDE the binding belongs in
the compositor and runs the CLI:

```
bindsym $mod+n exec syrinx toggle
```

The daemon logs which of these applies at startup rather than failing quietly,
the generated config says so too, and the GUI's help window reports what
actually happened — a hotkey can be configured and still not be listening,
because another application already owns the combination.

## Streaming to a file

`stream_to` appends the transcript as it is dictated, rather than saving only
at the end:

```toml
stream_to = "~/transcripts/notes.txt"
format = "timestamped"
```

Or per run: `syrinx start --stream notes.txt`. In the GUI, the **Stream…**
button picks a file, offering the current local time as the name —
`2026-08-21_14-59-41.txt`, the same default **Save as…** uses. Ordered
largest unit first, so sorting a folder by name sorts it by time.

Every session appends to the same file, so stopping and starting continues
where you left off, and a crash costs the last sentence rather than the whole
session — verified against `kill -9` mid-session.

Only committed text is written. A stamped line continues while speech does and
breaks on a pause, because the model splits on chunk boundaries rather than
words: a line per fragment produced "brown fox j" then "umps over the".

## Keys

Press **F1** in the GUI, or the Help button, for a window listing everything
that can drive syrinx: the in-window keys, the global hotkey and its real
status, what the tray icon does, and the CLI equivalents.

| Key | Action |
|---|---|
| `Ctrl+D` | Start or stop dictation |
| `Ctrl+S` | Save the transcript |
| `F1` | Show or hide the help window |

Every one takes a modifier: a bare key would fire while typing into the server
address field.

## Windows notes

Per-application capture is Linux-only. Windows has had process loopback since
10 2004, but cpal does not expose it, so it would mean hand-written WASAPI.
Capturing *all* system audio works on both.

`ffmpeg` must be on PATH to transcribe files. If winget installed it, PATH
points at a zero-length reparse point that cannot be spawned even though
`ffmpeg -version` works in a shell; put the real `bin` directory on PATH
instead.

### Reaching the server

The client is thin: no model, no GPU work, just audio out and text back. Name
the machine the server is on:

```toml
server = "192.168.1.10"     # or a hostname: laptop.lan
```

Just the host. Syrinx supplies the port and the endpoint, so there is nothing
to memorise and one thing to change when the server moves. Append a port
(`laptop.lan:9000`) if it is not on the default. A complete `url = "..."`
still works as an override, for a reverse proxy on a path syrinx would not
choose.

The server binds to `127.0.0.1` by default, which no other machine can reach.
For a remote client set `bind = "0.0.0.0:8770"` in the **server's** config.
Authentication is the shared `token` and nothing else -- there is no TLS -- so
this belongs on a trusted network only.

### Building

```powershell
winget install Rustlang.Rustup      # MSVC toolchain, plus VS Build Tools
cargo build --release -p syrinx-cli -p syrinx-gui
```

Build **release** for day-to-day use: only the release binary is linked as a
Windows-subsystem application, so only that one opens without a console window
behind it. Double-clicking `target\release\syrinx-gui.exe` starts the daemon
if none is running, opens the window, and leaves the daemon running when the
window is closed.

The daemon is spawned with `DETACHED_PROCESS`, so it has no console of its own
and closing the terminal you launched from cannot reach it. Running
`syrinx daemon` directly in a terminal is the ordinary foreground case and does
stop with that terminal; launch the GUI, or use the tray, to get one that
persists.

Anything that goes wrong before the window appears is shown in a message box.
A double-clicked application has no console to print to, so without that a bad
token would look identical to a broken download.

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

On Linux the fallback typing method is `wtype`, which uses the Wayland
virtual-keyboard protocol. It is correct for most native Wayland applications
but **fails in Electron and Chromium apps**: each call creates and destroys a virtual keyboard,
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

With the daemon running, the default `inject = "auto"` picks `ydotool` by
itself — see below — so no config change is needed. To pin it explicitly, the
whole file should look like this; `token` is required, so replacing the file
with only the one new line will stop it loading:

```toml
server = "127.0.0.1"
token = "your-shared-token"
inject = "ydotool"
```

Options are `auto` (default), `wtype`, `ydotool`, `paste`, and `sendinput`.
Paste copies to the
clipboard and sends Ctrl+V, restoring the previous clipboard afterwards — the
most broadly compatible of the three, though terminals need Ctrl+Shift+V so it
is a poor fit there.
