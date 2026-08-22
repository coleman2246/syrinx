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

    private var session: SyrinxCoreSession?
    private var capture: AudioCapture?
    private var poll: Timer?
    private var keyboardWatch: Timer?
    /// The last value the watcher acted on.
    ///
    /// The shared flag says what should be happening, not what changed, and
    /// it defaults to false. Acting on the value rather than on a change to
    /// it means a session started here is stopped half a second later by a
    /// flag nobody set -- which looks like the session silently failing,
    /// since stopping cleanly reports no error.
    private var lastWant = false

    /// Wire up whichever channel the keyboard will use.
    ///
    /// Both directions matter. Text has to reach the keyboard, and the
    /// keyboard's mic button has to reach the microphone -- which lives here,
    /// because an extension cannot open one.
    init() {
        if Handoff.usingSharedContainer {
            lastWant = Handoff.wantsCapture
            // A file-backed flag has no way to announce itself, so it is
            // polled. Twice a second: this is a button press, not speech.
            keyboardWatch = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) {
                [weak self] _ in
                Task { @MainActor in self?.checkKeyboardFlag() }
            }
        } else {
            LocalLink.shared.onCapture = { [weak self] wanted in
                Task { @MainActor in self?.followKeyboard(wanted) }
            }
            LocalLink.shared.startServing()
        }
        // A launch means nothing is capturing, whatever the flag survived the
        // last run saying. Left stale, a true from a killed session shows the
        // keyboard a live mic that is not recording anything.
        publishState()
    }

    func toggle() { running ? stop() : start() }

    /// Act only when the keyboard has actually changed the flag.
    private func checkKeyboardFlag() {
        let want = Handoff.wantsCapture
        guard want != lastWant else { return }
        lastWant = want
        followKeyboard(want)
    }

    /// Act on the keyboard's mic button.
    private func followKeyboard(_ wanted: Bool) {
        if wanted && !running { start() }
        if !wanted && running { stop() }
    }

    /// Tell the keyboard what is actually happening.
    ///
    /// The flag is written from both ends: the keyboard sets it to ask, and
    /// this sets it to answer. Without the answer its mic button reports a
    /// request rather than a state, and a start that failed still looks live.
    private func publishState() {
        lastWant = running
        Handoff.wantsCapture = running
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
            self.error = "Microphone: \(error.localizedDescription)"
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
        running = false
        status = "Idle"
    }

    func clear() { transcript = ""; error = nil }
}
