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
    /// Whether to stay resident so the keyboard can start dictation without
    /// the app being opened first. On by default: that is the whole point of
    /// the keyboard, and the cost is a track of silence.
    @AppStorage("keepAwake") var keepAwake = true

    private var session: SyrinxCoreSession?
    private var capture: AudioCapture?
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
        if keepAwake { KeepAwake.shared.start() }
    }

    func setKeepAwake(_ on: Bool) {
        keepAwake = on
        // Never while recording: capture holds the session itself, and the
        // app cannot be playing and recording under one category.
        if on && !running {
            KeepAwake.shared.start()
        } else if !on {
            KeepAwake.shared.stop()
            if !running { AudioSession.deactivate() }
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
        guard let s = SyrinxCoreSession(url: serverURL, token: token) else {
            error = "Could not start a session — check the server address."
            publishState()
            return
        }
        session = s

        // The capture callback runs on the audio thread and hands straight to
        // Rust. No hop to the main actor: that would add latency to every
        // buffer and risk dropping audio under UI load.
        capture = AudioCapture { [weak s] samples, count in
            s?.push(samples, count: count)
        }
        guard let c = capture else {
            error = "Could not build the audio pipeline."
            session = nil
            publishState()
            return
        }

        do {
            try c.start()
        } catch {
            self.error = "Microphone: \(AudioCapture.describe(error))"
            session = nil
            capture = nil
            publishState()
            return
        }

        running = true
        // 10 Hz is well under the rate fragments arrive at, so text appears
        // promptly without spinning the main thread.
        poll = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
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
        if let e = s.takeError() {
            error = e
            stop()
        }
    }

    func stop() {
        poll?.invalidate(); poll = nil
        capture?.stop(); capture = nil
        session?.stop(); session = nil
        // The session stays up so the keyboard can start the next one; giving
        // it back would mean reclaiming it from the background, which fails.
        if !keepAwake { AudioSession.deactivate() }
        running = false
        status = "Idle"
    }

    func clear() { transcript = ""; error = nil }
}
