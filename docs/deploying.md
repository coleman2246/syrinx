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

The 2.4 GB model directory is not in the image and not in the repository. Copy
it to the host once:

```bash
rsync -a --info=progress2 \
  ~/.local/share/parakeet-dictation/nemotron/ \
  gpu-host:/srv/syrinx/models/
```

Four files: `encoder.onnx`, `encoder.onnx.data`, `decoder_joint.onnx`,
`tokenizer.model`.

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
server = "wss://dictate.example.com"
token  = "the token from step 3"
```

On the LAN, `server = "192.168.1.10"` still works if the port is published
there.

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
