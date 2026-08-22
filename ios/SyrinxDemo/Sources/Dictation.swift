import Foundation
import SwiftUI

/// Ties capture to the core and publishes what the view needs.
///
/// Deliberately thin. Anything that could live in Rust does — this holds no
/// protocol knowledge, no reconnection policy and no transcript assembly
/// beyond appending what the core hands over.
extension Bundle {
    /// A build-time value from Info.plist, or nil when it was left empty --
    /// which is what a clean checkout without Local.xcconfig produces.
    static func baked(_ key: String) -> String? {
        guard let v = main.object(forInfoDictionaryKey: key) as? String,
              !v.isEmpty else { return nil }
        return v
    }
}

@MainActor
final class Dictation: ObservableObject {
    @Published var transcript = ""
    @Published var status = "Idle"
    @Published var error: String?
    @Published var running = false { didSet { publishState() } }

    // Defaults come from the build, so the app works on first launch without
    // anyone typing a 40-character token on a phone keyboard. Whatever the
    // user sets in Settings wins from then on.
    @AppStorage("serverURL") var serverURL = Bundle.baked("SyrinxURL")
        ?? "wss://dictate.example.com/v1/stream"
    @AppStorage("token") var token = Bundle.baked("SyrinxToken") ?? ""
    /// Whether to hold the microphone open so the keyboard can start
    /// dictation without the app being opened first. On by default: that is
    /// the whole point of the keyboard. The cost is the microphone indicator.
    @AppStorage("keepAwake") var holdMicrophone = true
    /// Which input to use, by UID. Empty means automatic, which prefers
    /// AirPods and avoids car microphones.
    @AppStorage("microphoneUID") var microphoneUID = ""

    private var session: SyrinxCoreSession?
    /// Opened once and kept, rather than built per dictation. Nothing in the
    /// start path may touch the audio system, because the start path runs in
    /// the background and that is precisely what iOS refuses.
    private let capture = AudioCapture()
    private var poll: Timer?

    /// Serve the keyboard.
    ///
    /// Always, not only when some entitlement is missing: the extension has
    /// exactly one way to reach this process, and a listener that is
    /// conditional on anything is a listener that is sometimes absent for a
    /// reason nobody can see from the other end.
    init() {
        LocalLink.shared.onCapture = { [weak self] wanted in
            Task { @MainActor in self?.followKeyboard(wanted) }
        }
        LocalLink.shared.startServing()

        // Permission, then the microphone, both while the app is still on
        // screen. Everything that iOS will not let a background app do has to
        // happen here or not at all.
        AudioCapture.requestPermission { [weak self] granted in
            Task { @MainActor in
                guard let self, granted, self.holdMicrophone else { return }
                self.openMicrophone()
            }
        }

        // AirPods connecting, or getting into a car, re-picks the input --
        // usually badly -- and leaves the tap built for the old one.
        NotificationCenter.default.addObserver(
            forName: .syrinxRouteChanged, object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.rebuildForNewInput() }
        }

        // A media services reset invalidates every audio object in the
        // process. Anything still running is running on rubble.
        NotificationCenter.default.addObserver(
            forName: .syrinxAudioReset, object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                if self.running {
                    self.stop()
                    self.error = "The audio system restarted. Start again."
                }
                self.capture?.close()
                if self.holdMicrophone { self.openMicrophone() }
            }
        }
    }

    /// Open the microphone, reporting rather than swallowing a refusal.
    private func openMicrophone() {
        guard let c = capture else {
            error = "Could not build the audio pipeline."
            return
        }
        AudioSession.selectInput(uid: microphoneUID.isEmpty ? nil : microphoneUID)
        do {
            try c.open()
        } catch {
            self.error = "Microphone: \(AudioCapture.describe(error))"
        }
    }

    /// Pin an input, or pass nil for automatic.
    func setMicrophone(uid: String?) {
        microphoneUID = uid ?? ""
        rebuildForNewInput()
    }

    /// Follow the input to wherever it went.
    private func rebuildForNewInput() {
        guard let c = capture, c.isOpen else {
            AudioSession.selectInput(uid: microphoneUID.isEmpty ? nil : microphoneUID)
            return
        }
        AudioSession.selectInput(uid: microphoneUID.isEmpty ? nil : microphoneUID)
        do {
            try c.reopen()
        } catch {
            // Worth surfacing: capture is now dead, and silently dead audio is
            // exactly the failure that is impossible to diagnose from outside.
            self.error = "Microphone: \(AudioCapture.describe(error))"
        }
    }

    func setHoldMicrophone(_ on: Bool) {
        holdMicrophone = on
        if on {
            openMicrophone()
        } else if !running {
            capture?.close()
            AudioSession.deactivate()
        }
    }

    func toggle() { running ? stop() : start() }

    /// Act on the keyboard's mic button. It cannot open a microphone, so this
    /// side does it.
    private func followKeyboard(_ wanted: Bool) {
        if wanted && !running { start() }
        if !wanted && running { stop() }
    }

    /// Tell the keyboard what is actually happening, including when a start
    /// failed -- otherwise its mic button reports a request rather than a
    /// state, and a session that never began still looks live.
    private func publishState() {
        LocalLink.shared.setCapturing(running)
    }

    func start() {
        error = nil
        AudioCapture.requestPermission { [weak self] granted in
            guard let self else { return }
            guard granted else {
                self.error = "Microphone access was refused. Settings › Privacy › Microphone."
                self.publishState()
                return
            }
            self.begin()
        }
    }

    private func begin() {
        guard let c = capture, c.isOpen else {
            // Only reachable with the microphone released: held open, this
            // path cannot fail, which is the entire point of holding it.
            openMicrophone()
            guard let c = capture, c.isOpen else {
                publishState()
                return
            }
            begin(with: c)
            return
        }
        begin(with: c)
    }

    private func begin(with c: AudioCapture) {
        guard let s = SyrinxCoreSession(url: serverURL, token: token) else {
            error = "Could not start a session — check the server address."
            publishState()
            return
        }
        session = s

        // Runs on the audio thread and hands straight to Rust. No hop to the
        // main actor: that would add latency to every buffer and risk dropping
        // audio under UI load.
        c.deliver { [weak s] samples, count in
            s?.push(samples, count: count)
        }

        running = true
        // 20 Hz. Text does not arrive nearly this fast, but the spectrum does,
        // and the keyboard cannot show a level this side has not collected.
        poll = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.drain() }
        }
    }

    private func drain() {
        guard let s = session else { return }
        if let fresh = s.takeText() {
            transcript += fresh
            // Hand it to the keyboard extension, which cannot record but can
            // type. Published as it arrives so insertion tracks speech.
            Handoff.publish(fresh)
        }
        status = s.status.label
        LocalLink.shared.setLevels(s.levels())
        if let e = s.takeError() {
            error = e
            stop()
        }
    }

    func stop() {
        poll?.invalidate(); poll = nil
        capture?.deliver(to: nil)
        session?.stop(); session = nil
        // The microphone stays open so the keyboard can start the next one;
        // reopening it would mean claiming it from the background, which is
        // the thing iOS refuses.
        if !holdMicrophone {
            capture?.close()
            AudioSession.deactivate()
        }
        running = false
        status = "Idle"
    }

    func clear() { transcript = ""; error = nil }
}
