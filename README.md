# syrinx

Self-hosted streaming speech-to-text, built on NVIDIA Parakeet/Nemotron
cache-aware streaming ASR via [`parakeet-rs`](https://github.com/altunenes/parakeet-rs).

Unlike whisper (which must see a complete utterance before emitting anything),
a transducer emits tokens as audio arrives. That gives word-level latency without
Vosk's accuracy penalty: ~1.93% WER on librispeech test-clean, with punctuation.

## The model

Speech recognition is [NVIDIA Nemotron Speech Streaming EN
0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) — a
600M-parameter cache-aware FastConformer-RNNT, English, with punctuation and
capitalisation. Being a transducer is why this streams at all: it emits tokens
as audio arrives rather than needing a complete segment.

It runs through [`parakeet-rs`](https://github.com/altunenes/parakeet-rs), and
the ONNX export syrinx uses is the one that crate publishes:

```bash
MODEL=https://huggingface.co/altunenes/parakeet-rs/resolve/main/nemotron-speech-streaming-en-0.6b
mkdir -p ~/models/nemotron && cd ~/models/nemotron
for f in encoder.onnx encoder.onnx.data decoder_joint.onnx tokenizer.model; do
  curl -fL# -O "$MODEL/$f"
done
```

Or, with the HuggingFace CLI if you have it:

```bash
hf download altunenes/parakeet-rs \
  --include "nemotron-speech-streaming-en-0.6b/*" --local-dir ~/models
```

That gives four files totalling 2.4 GB — `encoder.onnx`, `encoder.onnx.data`,
`decoder_joint.onnx`, `tokenizer.model` — and the directory containing them is
what `model_dir` points at. They are not in this repository and not in the
container image: they are large, and they change on a different schedule from
the code.

**Licence:** the model is NVIDIA's, under the [NVIDIA Open Model License
Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
which permits commercial use. That is separate from the licence on this code —
read it before shipping anything built on top.

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
  the buffer, so the server may revise it. The only mode that can carry speaker
  labels; see **Speaker labels** below.
- **bulk** — offline file transcription over HTTP, using the more accurate
  non-streaming model.

## Speaker labels

In transcribe mode the server can tell voices apart and tag each turn
**Speaker 1**, **Speaker 2**, and so on — enough to read back a meeting and
follow who said what. The labels are anonymous by construction: they come from
the sound of a voice, and nothing here ever learns a name. Putting names to
them is a job for an LLM given the finished transcript, which does it from
what people say about each other and is far better at it than anything
acoustic.

Off at both ends by default. The server needs two more ONNX models, 29 MB
together:

```bash
mkdir -p ~/models/diarize && cd ~/models/diarize

# Voice activity detection -- is anyone speaking? 2.2 MB, MIT.
curl -fL# -o silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/v6.2.1/src/silero_vad/data/silero_vad.onnx

# Speaker embeddings -- are these two stretches the same voice? 26.5 MB,
# Apache-2.0.
curl -fL# -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx
```

`speaker-recongition-models` is spelled that way upstream. It is their typo,
it is load-bearing in the URL, and correcting it gets a 404.

The **server** needs two things, and one without the other does nothing: the
directory in its config, and the feature compiled in. A binary built without
it ignores the models entirely, and a plain `cargo build` produces exactly
that binary.

```toml
diarize_model_dir = "/home/you/models/diarize"
```

```bash
cargo build --release -p syrinx-server --features diarize   # or cuda,diarize
```

Spelled out in full: the shell expanded the `~` in the download above, and the
server will not — it takes `model_dir` and `diarize_model_dir` exactly as
written. Nor is there an environment override for this one, unlike the token
and the model directory. It decides how the service behaves rather than where
it is deployed, and those settings belong in a file somebody can review.

Then, in the **client's** config:

```toml
mode = "transcribe"
diarize = true
```

### Asking is not receiving

`diarize = true` is a request, and the handshake answers it honestly rather
than refusing the session. Two things on the server can turn it down: it was
built without the feature, or it could not read the models. Either way the
session runs on unlabelled, because a transcript without speaker labels is
worth incomparably more than no transcript at all. The GUI says so under the
status line, once a session is running — *Speaker labels unavailable on this
server* — and the server logs which of the two it was at `error!`, being the
only end that knows.

**`type` and `both` never ask in the first place**, so none of that applies to
them and no notice appears. Labels are there to be read beside the words, and
these modes put the words at the cursor, where a `Speaker 2:` prefix would
land in the middle of whatever you are working in.

Worth knowing in `both`, which does keep a transcript: that transcript will be
unlabelled however the server is configured, and nothing in the interface will
say why, because nothing went wrong. `transcribe` is the mode that asks.

### What to expect

The labels are stable, which is the property that matters: over an 87-minute
meeting every speaker kept the label they started with — no renumbering, no
drift. A transcript whose Speaker 3 stops meaning the same person halfway
down is worse than one with no labels at all.

Short interjections arrive unlabelled, and that is the design rather than a
gap in it. When the diarizer has not heard enough of a voice to be sure, it
says nothing instead of attributing "yeah" or "okay" to whoever was already
talking. The start of a session goes the same way: a voice has to put together
close to four seconds of speech, across four windows that agree with each
other, before it is given a label at all — and that is four seconds of
talking, not four seconds of meeting. Unlabelled text still appears, joining
the turn already open, and a label trails its text by about a second, which is
the server holding two chunks back so the diarizer can catch up.

Both of those numbers are settings, in case your meetings disagree with the
recordings they were measured on:

```toml
diarize_lag_chunks = 2    # chunks a commit waits for its label
diarize_min_pool = 4      # agreeing windows before a new speaker is minted
```

Dropping the lag to 1 hands text over half a second sooner and pays for it at
the start of a turn, where the label is still the previous speaker's; 0 turns
the wait off altogether. Dropping the pool to 3 gives a quiet participant a
number three quarters of a second earlier, and risks the split this whole
design is arranged to avoid — on the 87-minute meeting a pool of 3 found six
speakers in a room of five, and a pool of 2 found twenty in a room of four.
Both keys are optional, both are refused at startup if they are wildly out of
range, and neither is environment-overridable, for the same reason
`diarize_model_dir` is not.

What has *not* been measured is the caveat worth carrying. No split (one
person acquiring a second label) and no merge (two people sharing one)
occurred on any meeting tested — but those were AMI recordings made through
close-talking headset microphones, which are cleaner than a conference call
arriving over a laptop speakerphone. Expect worse on worse audio. The
embedding model is English, trained on VoxCeleb, and overlapping speech is
given to one of the speakers rather than separated.

See [`docs/specs/`](docs/specs/) for what was measured, on which meetings, and
what those numbers do not establish.

**Licences:** silero-vad is MIT. The embedding model is Apache-2.0, from the
[3D-Speaker](https://github.com/modelscope/3D-Speaker) project, distributed
through the sherpa-onnx model zoo. Both permit commercial use, both are
separate from the licence on this code, and — like the ASR model — neither is
in this repository nor in the container image.

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

The image is built with speaker labelling compiled in, and it stays dormant
until a `diarize` subdirectory of that mount holds the two models and
`docker/config.toml` names it. See [docs/deploying.md](docs/deploying.md).

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
url = "wss://dictate.example.com/v1/stream"
```

No port, because `wss://` means 443. The client validates the certificate chain
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
url = "ws://192.168.1.10:8770/v1/stream"
```

Written out in full and used exactly as given: scheme, host, port, path.
Nothing is inferred.

syrinx used to accept a bare host and build the URL around it. That is
convenient right up to the first address it gets wrong — a bare
`dictate.example.com` became `ws://dictate.example.com:8770/v1/stream`, which
is plaintext on the wrong port when what you meant was `wss://` on 443, and
there was no way to say so. A setting you cannot override is worse than one
you have to type.

`server = "..."` is accepted as a name for the same setting, so older configs
keep working.

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
url = "ws://127.0.0.1:8770/v1/stream"
token = "your-shared-token"
inject = "ydotool"
```

Options are `auto` (default), `wtype`, `ydotool`, `paste`, and `sendinput`.
Paste copies to the
clipboard and sends Ctrl+V, restoring the previous clipboard afterwards — the
most broadly compatible of the three, though terminals need Ctrl+Shift+V so it
is a poor fit there.

## iPhone

An app and a custom keyboard, dictating into any app that takes text. The app
captures and transcribes with the same Rust client the desktop uses; the
keyboard types the result at the cursor. A keyboard extension cannot open a
microphone on any iOS version, which is why it is two pieces rather than one.

Building needs a Mac, since Xcode runs nowhere else:

```sh
make ios-framework   # only when anything under crates/ changed
make ios             # builds on the VM and copies the .ipa back
```

Those targets drive a macOS VM over SSH; `ios/macos-vm.sh` launches one. Apple's
licence permits macOS only on Apple hardware, so that route is a decision to
make deliberately rather than a supported path — a Mac you own needs neither
the VM nor the two `make` targets, just the three scripts in `ios/`.

The build is unsigned; a sideloader signs it with your Apple ID. After
installing, turn on **Full Access** for the keyboard and open the app once —
iOS will not let a background app claim a microphone it did not already have,
so the app holds one from launch. That is also why the microphone indicator
stays lit while it is resident.

See [docs/ios.md](docs/ios.md) for the architecture, the build VM, and the
platform constraints that shaped both — several of them are the opposite of
what the error messages suggest.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).

Every dependency is permissive — MIT, Apache-2.0, BSD, ISC, Zlib, Unicode-3.0
or Unlicense — so all of them are GPL-3 compatible. Checked rather than
assumed: nothing in the tree needed a second look.

**The models are licensed separately and are not covered by this.** The ASR
model is NVIDIA's, under the [NVIDIA Open Model License
Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
which permits commercial use. The two optional speaker-labelling models are MIT
(silero-vad) and Apache-2.0 (3D-Speaker ERes2Net). No weights are in this
repository or in the container image, so nothing here redistributes any of
them.
