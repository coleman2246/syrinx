import AVFoundation

/// The one owner of the process's audio session.
///
/// iOS refuses to *activate* an audio session from the background. Category
/// options change which refusal you get, not whether you get one: a mixable
/// `.playAndRecord` session is turned away exactly as a non-mixable `.record`
/// one is, with '!int'.
///
/// So activation happens once at launch, while the app is on screen and
/// allowed to, and the session is held for the life of the process. Starting
/// capture later only starts an engine on a session that is already live,
/// which iOS does permit.
///
/// The cost is real: a held `.playAndRecord` session keeps the microphone
/// indicator lit whenever the app is resident. That is what iOS charges for
/// letting a background app have a microphone, and it is what the "Keep
/// Syrinx awake" switch buys.
enum AudioSession {
    /// What iOS last did to the session without being asked.
    ///
    /// The session can be taken away underneath us -- an interruption, or the
    /// audio server restarting -- and nothing tells the code that made the
    /// original request. Recorded because the resulting failure is a bare
    /// 'what' at the next call, which explains nothing about when it broke.
    private(set) static var lastEvent = "none"
    private(set) static var eventCount = 0

    /// Bring the session up, or confirm it is already up.
    ///
    /// `setActive(true)` on a session that is already active is a no-op and
    /// does not throw, so this makes no attempt to remember whether it has
    /// been called. It used to, and a cached "yes" that iOS had already
    /// invalidated meant skipping the one call that would have fixed it.
    static func ensureActive() throws {
        observe()
        let s = AVAudioSession.sharedInstance()
        if s.category != .playAndRecord || !s.categoryOptions.contains(.mixWithOthers) {
            // .measurement disables the input processing that would fight the
            // recogniser. .mixWithOthers is not accepted on .record, which is
            // why the category is .playAndRecord despite nothing being played.
            try labelled("setting the audio category") {
                try s.setCategory(.playAndRecord, mode: .measurement,
                                  options: [.mixWithOthers, .allowBluetooth])
            }
        }
        try labelled("activating the audio session") {
            try s.setActive(true, options: [])
        }
    }

    /// Give the session up, which also clears the microphone indicator.
    static func deactivate() {
        try? AVAudioSession.sharedInstance()
            .setActive(false, options: [.notifyOthersOnDeactivation])
    }

    /// Name the step that failed.
    ///
    /// Two calls here can throw and both report the same opaque code, so an
    /// unlabelled failure leaves no way to tell "iOS refused the session" from
    /// "the session was fine and the engine would not start".
    private static func labelled(_ step: String, _ body: () throws -> Void) throws {
        do {
            try body()
        } catch {
            throw NSError(
                domain: "Syrinx.audio",
                code: (error as NSError).code,
                userInfo: [NSLocalizedDescriptionKey: "\(step): \(describe(error))"])
        }
    }

    /// Decode CoreAudio's four-character codes, which it prints in decimal.
    static func describe(_ error: Error) -> String {
        let ns = error as NSError
        if ns.domain.hasPrefix("Syrinx") { return ns.localizedDescription }
        guard let code = fourCharCode(ns.code) else { return error.localizedDescription }
        switch code {
        case "!int":
            return "iOS would not start the microphone from the background ('!int'). "
                + "Open Syrinx with \"Keep Syrinx awake\" on, so the session is held "
                + "rather than claimed while backgrounded."
        case "what":
            return "the audio system rejected the request without saying why ('what'). "
                + "This usually means the session was torn down since it was last used."
        case "!pri": return "another app is holding the microphone ('!pri')."
        case "!ini": return "the audio session was not ready ('!ini')."
        case "!dev": return "no microphone is available ('!dev')."
        default:     return "\(error.localizedDescription) ('\(code)')"
        }
    }

    private static func fourCharCode(_ code: Int) -> String? {
        let v = UInt32(bitPattern: Int32(truncatingIfNeeded: code))
        let bytes = [UInt8(truncatingIfNeeded: v >> 24), UInt8(truncatingIfNeeded: v >> 16),
                     UInt8(truncatingIfNeeded: v >> 8), UInt8(truncatingIfNeeded: v)]
        guard bytes.allSatisfy({ (0x20...0x7e).contains($0) }) else { return nil }
        return String(bytes: bytes, encoding: .ascii)
    }

    private static var observing = false

    /// Watch for the session being taken away.
    ///
    /// A media services reset invalidates every audio object in the process,
    /// and every later call fails with 'what' until they are rebuilt. Nothing
    /// else in the app can distinguish that from a bug, so it is recorded and
    /// the session is reclaimed.
    private static func observe() {
        guard !observing else { return }
        observing = true
        let centre = NotificationCenter.default

        centre.addObserver(forName: AVAudioSession.interruptionNotification,
                           object: nil, queue: .main) { note in
            let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt ?? 0
            let ended = AVAudioSession.InterruptionType(rawValue: raw) == .ended
            record(ended ? "interruption ended" : "interrupted")
            if ended { try? ensureActive() }
        }
        centre.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification,
                           object: nil, queue: .main) { _ in
            record("media services reset")
            try? ensureActive()
            NotificationCenter.default.post(name: .syrinxAudioReset, object: nil)
        }
        centre.addObserver(forName: AVAudioSession.routeChangeNotification,
                           object: nil, queue: .main) { _ in
            record("route change")
        }
    }

    private static func record(_ what: String) {
        lastEvent = what
        eventCount += 1
    }

    static var diagnostics: String {
        let s = AVAudioSession.sharedInstance()
        let permission: String
        if #available(iOS 17.0, *) {
            permission = "\(AVAudioApplication.shared.recordPermission)"
        } else {
            permission = "\(s.recordPermission)"
        }
        return """
        category: \(s.category.rawValue.replacingOccurrences(of: "AVAudioSessionCategory", with: ""))
        options: \(s.categoryOptions.rawValue)
        mic permission: \(permission)
        other audio: \(s.isOtherAudioPlaying)
        inputs: \(s.currentRoute.inputs.map(\.portType.rawValue).joined(separator: ",")) 
        last event: \(lastEvent) (\(eventCount) total)
        """
    }
}

extension Notification.Name {
    /// Everything audio has to be rebuilt.
    static let syrinxAudioReset = Notification.Name("space.aragonite.syrinx.audioReset")
}
