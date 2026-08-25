# Deploying to a GPU server

Written for a headless Ubuntu host with an NVIDIA card, Docker, and an nginx
that already terminates TLS. Adjust the names; nothing here is specific to one
machine.

## 1. Get the code

```bash
git clone https://github.com/coleman2246/syrinx.git
cd syrinx
```

Deploying is `git pull` and rebuild, so the server runs a commit you can name
rather than an image someone built by hand.

## 2. Models

The 2.4 GB model directory is not in the image and not in the repository.
Either download it on the host:

```bash
MODEL=https://huggingface.co/altunenes/parakeet-rs/resolve/main/nemotron-speech-streaming-en-0.6b
sudo mkdir -p /srv/syrinx/models && cd /srv/syrinx/models
for f in encoder.onnx encoder.onnx.data decoder_joint.onnx tokenizer.model; do
  sudo curl -fL# -O "$MODEL/$f"
done
```

`curl` rather than the HuggingFace CLI so a headless server needs no Python
toolchain for this.

or copy one you already have:

```bash
rsync -a --info=progress2 ~/models/nemotron-.../ gpu-host:/srv/syrinx/models/
```

Four files: `encoder.onnx`, `encoder.onnx.data`, `decoder_joint.onnx`,
`tokenizer.model`. See the README for what the model is and its licence.

### Speaker labels (optional)

Skip this unless you want meeting transcripts tagged Speaker 1, Speaker 2 —
see **Speaker labels** in the README for what they are and what to expect. Two
more models, 29 MB, in a `diarize` subdirectory of the one you just filled:

```bash
sudo mkdir -p /srv/syrinx/models/diarize && cd /srv/syrinx/models/diarize
sudo curl -fL# -o silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/v6.2.1/src/silero_vad/data/silero_vad.onnx
sudo curl -fL# -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx
```

A subdirectory rather than a second volume: the mount is already there, it is
already read-only, and the ASR loader opens its four files by name and never
notices a neighbour. `docker/compose.yaml` carries the second-mount line for
anyone who keeps these somewhere else.

Then uncomment one line in `docker/config.toml`:

```toml
diarize_model_dir = "/models/diarize"
```

That line cannot come from `docker/.env`: only the settings that vary per
deployment are environment-overridable, and this one says how the service
behaves. Which means enabling speaker labels edits a file this repository
tracks — the one wrinkle in the container story. It rebuilds in seconds
(`config.toml` is copied in the last layers, so no Rust is recompiled), and
`git stash` before a `git pull` if the edit ever conflicts. If you would rather
not carry a local edit at all, bind-mount your own file over
`/etc/syrinx/config.toml` instead.

Two optional keys live in the same file, for tuning against the meetings this
server actually sees: `diarize_lag_chunks` (2) is how many 560 ms chunks a
commit waits for its speaker label, and `diarize_min_pool` (4) is how many
agreeing 1.5 s windows it takes to mint a new speaker. Lower is quicker and
less sure in both cases — a shorter wait costs attribution at the start of a
turn, a smaller pool risks splitting one person across two labels. **Speaker
labels** in the README has the measurements behind the defaults. Unlike a
model file, a value out of range here does stop the server: it is a typo in a
config the operator just edited, and the failure that helps is the one at
startup with the key named in it.

The image is built with `--features cuda,diarize`, so nothing else changes: no
new runtime dependency — the diarizer drives the same ONNX Runtime the ASR
already loads — and about 6% of one core per session on top of the ASR.

Check the startup log either way. It says `speaker labelling available` with
the two paths if it worked, and an `error!` line naming the file it could not
read if it did not; it never refuses to start over this.

## 3. Configure

```bash
cp docker/.env.example docker/.env
```

Fill in:

- `SYRINX_TOKEN` — `openssl rand -hex 24`. Not a memorable word: on anything
  internet-facing this is the only thing between a stranger and a transcription
  service on your GPU.
- `SYRINX_MODELS` — where step 2 put them.
- `SYRINX_LISTEN=127.0.0.1:8770` — **when nginx is terminating TLS**. Only the
  proxy should reach the plaintext port; the internet reaches 443.

`docker/.env` is gitignored. It is the one file that must never be committed.

## 4. Build and run

```bash
docker compose -f docker/compose.yaml up -d --build
docker compose -f docker/compose.yaml ps      # (healthy)
```

The build takes several minutes: it compiles the server and downloads ONNX
Runtime. It restarts on boot.

## 5. TLS

Point a name at the host, then issue a certificate with whatever the host
already uses:

```bash
sudo certbot certonly --webroot -w /var/www/certbot -d dictate.example.com
```

Add the server block from `docker/nginx-syrinx.conf.example`, then:

```bash
sudo nginx -t && sudo systemctl reload nginx     # or reload the container
```

`nginx -t` first, every time. A syntax error takes down every other site the
proxy serves, not just this one.

If nothing else is on 80 and 443, skip nginx entirely and use
`docker/compose.tls.yaml`, which runs Caddy and handles certificates itself.

## 6. Point the clients at it

```toml
url   = "wss://dictate.example.com/v1/stream"
token = "the token from step 3"
```

The address is used exactly as written — no port, because `wss://` means 443.
On the LAN it would be `url = "ws://192.168.1.10:8770/v1/stream"`, if the port
is published there.

## Updating

```bash
cd syrinx && git pull
docker compose -f docker/compose.yaml up -d --build
```

## Sharing a GPU

syrinx is built to be the lowest-priority tenant. It holds no model when idle,
and refuses to load when doing so would leave less than `vram_floor_mib` free —
so a busy transcoder or a camera detector makes syrinx return `capacity` rather
than making them fail. That refusal is the design working, not a fault.

Check what else is on the card:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```
