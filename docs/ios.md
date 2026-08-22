# The iOS app and keyboard

Dictation on a phone, against your own server, into any app that takes text.

Two targets. **SyrinxDemo** is a small app that captures audio, runs the same
Rust client the desktop uses, and shows the transcript. **SyrinxKeyboard** is a
custom keyboard that types what the app transcribes at the cursor. The split is
forced: a keyboard extension cannot open a microphone, on any iOS version.

## How it fits together

```
microphone ─► AVAudioEngine ─► converter ─► syrinx-ios (C ABI) ─► syrinx-client ─► server
                                                                       │
                                              transcript ◄─────────────┘
                                                   │
                                    loopback 127.0.0.1:47632
                                                   │
                                  SyrinxKeyboard ─► insertText at the cursor
```

Everything below the microphone is the same Rust that runs on Linux and
Windows: protocol, WebSocket, TLS, streaming state, reconnection. `crates/syrinx-ios`
is a thin C ABI over it, and `SessionOptions::external_audio` is the seam that
lets Swift supply audio instead of the session opening a device itself.

The app and the keyboard talk over a loopback socket. That is not a fallback;
it is the only channel. See *Why loopback* below.

## Building

You need a Mac, or a macOS VM. There is one on this machine — see *The build
VM*. From a checkout on that machine:

```sh
./ios/generate.sh                      # project.yml -> SyrinxDemo.xcodeproj
./crates/syrinx-ios/build-xcframework.sh   # only when Rust changed
./ios/build-ipa.sh                     # -> ios/build/SyrinxDemo.ipa
```

`build-ipa.sh` does **not** rebuild the Rust framework. Changing anything under
`crates/` and running only `build-ipa.sh` links the previous static library and
silently produces an app without your change.

From Linux, `make ios` drives all three over SSH and copies the `.ipa` back.

`ios/Local.xcconfig` holds the server URL and token, is gitignored, and is baked
into the app at build time so the app works on first launch without anyone
typing a 40-character token on a phone. Copy `Local.xcconfig.example` and fill
it in. Note that xcconfig treats `//` as a comment *anywhere in a value*, which
eats the slashes in a URL — the example works around it with a `SLASH` variable
and there is no neater way.

## Installing

The build is unsigned (`CODE_SIGNING_ALLOWED=NO`), so a sideloader signs it with
your own Apple ID. Any of AltStore, SideStore or iLoader will do. Whatever signs
it must stay installed: a free certificate expires every seven days and
something has to refresh it.

After installing, two things are required and neither is optional:

1. **Settings › General › Keyboard › Keyboards › add Syrinx, then turn on Full
   Access.** Without it the extension has no network, so it cannot reach the
   app at all. The keyboard says so rather than failing silently.
2. **Open Syrinx once.** It holds the microphone from then on, which is what
   lets the keyboard start dictation later. See *Why nothing starts in the
   background*.

## Using it

Open the app once. Switch to anything with a text field, choose the Syrinx
keyboard, tap the microphone. Speak; text appears at the cursor. The bars show
the same spectrum the desktop overlay draws, and the caption shows the last few
words typed — a meter can bounce happily while the transcript stays empty, so
both are needed to tell "hearing you" from "recognising you".

The **ⓘ** button types a diagnostic report into whatever field has focus. A
sideloaded extension has no console and no debugger, so this is the only way to
see what it thinks is happening. The app's half is in Settings › Keyboard.

## Constraints worth knowing before changing anything

These each cost hours to find. None of them is discoverable from the error
message you get.

### Why nothing starts in the background

iOS does not forbid a background app from recording. It forbids it from
*beginning* to. Activating a session, setting a category, and starting an
engine are all beginnings, and each is refused with `'!int'`
(`AVAudioSessionErrorCodeCannotInterruptOthers`, which CoreAudio prints as the
decimal `560557684`).

Choosing a different category does not help; it only moves which call gets
refused. Three separate attempts went that way before the pattern was clear.

So `AudioCapture` opens the session, the engine and the tap **at launch, in the
foreground**, and never closes them. Starting dictation sets a closure on the
running tap; stopping it clears one. There is no audio call left in that path.

The cost is that the microphone indicator stays lit for as long as the app is
resident, because the microphone genuinely is open. That is what iOS charges
for the feature, and the "Hold the microphone" setting is where the user
decides whether to pay it.

A running engine also keeps the app alive, so no silent-audio keep-alive trick
is needed. One existed; it is gone.

### Why loopback, and not an App Group

The obvious channel between an app and its extension is a shared container. It
was tried and removed.

A sideloader may grant the App Groups entitlement to one binary and not the
other. When that happened, the app wrote to a shared container and the keyboard
read from a socket, each reporting itself perfectly healthy. The container only
justified that risk if it worked somewhere loopback does not — and it does not,
because a keyboard needs Full Access for either one.

So there is one channel and nothing to disagree about. `LocalLinkProtocol.secret`
is a constant in a public repository and protects nothing from anyone who
looks; any app on the device can reach a loopback port. It stops unrelated
software stumbling into it, and that is all it is for.

The clipboard was also tried, and cannot work: iOS prompts before one process
reads what another wrote, and rewriting the user's clipboard several times a
second is hostile even when it succeeds.

### Why the keyboard cannot just launch the app

It has no `openURL` — `extensionContext.open` is unavailable to keyboards, and
the responder-chain workaround has been steadily hardened. More fundamentally,
iOS cannot launch an app *into the background*, so even a successful launch
would foreground Syrinx and throw the user out of the field they were typing
in. Holding the microphone is the answer instead.

### Microphone selection

iOS re-picks the input whenever something connects, and in a car it picks the
car, whose hands-free microphone is much worse than the phone's own. So the
choice is made in `AudioSession.rank`: AirPods, then a wired headset, then the
built-in microphone, and a car kit never automatically.

Telling AirPods from a car kit is a heuristic. Both are `.bluetoothHFP` and iOS
offers nothing but the device name to separate them, so the name is matched
against "airpods" and "beats". An unrecognised Bluetooth device therefore loses
to the built-in microphone rather than beating it, because a car microphone is
the more costly mistake. Anything can be pinned by hand in Settings.

A route change rebuilds the engine: the tap is installed for one specific
hardware format and a new input will not match it.

### Two build-system traps

`actool` refuses to compile an asset catalogue unless a simulator runtime is
installed, reporting *"No available simulator runtimes for platform
iphonesimulator"* — during a **device** build. The app icon therefore ships as
loose `CFBundleIconFiles` PNGs, which predates catalogues and iOS still
honours. Do not "fix" this by reintroducing an `.xcassets`.

`rustup` puts `~/.cargo/bin` on `PATH` from a shell profile, and a
non-interactive SSH reads none of them. `build-xcframework.sh` sources
`~/.cargo/env` itself; without that, remote builds fail with `cargo: command
not found` while cargo works fine when you log in and try it by hand.

### TLS

`rustls-native-certs` dispatches on `target_os = "macos"`, and iOS is `"ios"`:
Unix but not macOS, so it takes the Unix path, probes `/etc/ssl/certs`, and
finds nothing. `crates/syrinx-client/Cargo.toml` therefore selects
`rustls-tls-webpki-roots` for iOS and native roots everywhere else. A session
that connects on desktop and fails only on the phone is this.

## The build VM

A macOS Sequoia guest under QEMU/KVM (OSX-KVM), on this machine:

- Working directory `/mnt/winssd/macos-vm`, disk `mac_hdd_ng.img`, on the NTFS
  drive shared with Windows. Linux will refuse to mount that read-write while
  Windows is hibernated; `ntfsfix -d` clears the dirty flag.
- `-cpu Skylake-Client` is required. Sequoia panics into a boot loop on
  `Penryn`, which is what most OSX-KVM examples still use.
- SSH is forwarded to **`127.0.0.1:2222` only** — deliberately not `0.0.0.0`,
  which would expose the guest to the LAN.
- Key at `~/.ssh/macvm`. `ssh -i ~/.ssh/macvm -p 2222 cole@127.0.0.1`.
- Xcode 26.3. Xcode 26.6 requires macOS Tahoe 26.2 and will not run here.
- x86_64, and no simulator runtimes are installed. The Simulator is not useful
  anyway: it grants App Groups freely and does not enforce sideload signing, so
  it cannot reproduce the failures that actually happen on device. The
  xcframework's simulator slice is `aarch64-apple-ios-sim` and would not run
  here regardless.

Start it with the OSX-KVM launch script in that directory before building.
