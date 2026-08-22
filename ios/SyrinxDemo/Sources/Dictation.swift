import Foundation
import SwiftUI

/// Ties capture to the core and publishes what the view needs.
///
/// Deliberately thin. Anything that could live in Rust does — this holds no
/// protocol knowledge, no reconnection policy and no transcript assembly
/// beyond appending what the core hands over.
@MainActor
final class Dictation: ObservableObject {
    @Published var transcript = ""
    @Published var status = "Idle"
    @Published var error: String?
    @Published var running = false

    @AppStorage("serverURL") var serverURL = "wss://dictate.example.com/v1/stream"
    @AppStorage("token") var token = ""

    private var session: SyrinxCoreSession?
    private var capture: AudioCapture?
    private var poll: Timer?

    func toggle() { running ? stop() : start() }

    func start() {
        error = nil
        AudioCapture.requestPermission { [weak self] granted in
            guard let self else { return }
            guard granted else {
                self.error = "Microphone access was refused. Settings › Privacy › Microphone."
                return
            }
            self.begin()
        }
    }

    private func begin() {
        guard let s = SyrinxCoreSession(url: serverURL, token: token) else {
            error = "Could not start a session — check the server address."
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
            return
        }

        do {
            try c.start()
        } catch {
            self.error = "Microphone: \(error.localizedDescription)"
            session = nil
            capture = nil
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
        if let fresh = s.takeText() { transcript += fresh }
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
