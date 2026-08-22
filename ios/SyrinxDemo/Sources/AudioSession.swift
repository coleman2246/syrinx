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
        // Once. Re-setting the category in the background counts as claiming
        // the session afresh and is refused with '!int' -- which is how a
        // mode fallback, added to work around a different failure, turned
        // into this one.
        if !configured {
            // .measurement disables the input processing that would fight the
            // recogniser. .mixWithOthers is not accepted on .record, which is
            // why the category is .playAndRecord despite nothing being played.
            try labelled("setting the audio category") {
                try s.setCategory(.playAndRecord, mode: .measurement,
                                  options: [.mixWithOthers, .allowBluetooth])
            }
            configured = true
        }
        try labelled("activating the audio session") {
            try s.setActive(true, options: [])
        }
    }

    /// The inputs iOS is currently offering, best first.
    static var inputs: [AVAudioSessionPortDescription] {
        (AVAudioSession.sharedInstance().availableInputs ?? [])
            .sorted { rank($0) > rank($1) }
    }

    /// The input in use right now.
    static var currentInput: AVAudioSessionPortDescription? {
        AVAudioSession.sharedInstance().currentRoute.inputs.first
    }

    /// Choose an input: the one with `uid`, or the best available.
    ///
    /// Reapplied on every route change, because a preference expressed once
    /// is not a preference iOS remembers -- plugging in headphones or getting
    /// into a car re-picks the input, and the pick is often wrong.
    static func selectInput(uid: String?) {
        let available = inputs
        let wanted = uid.flatMap { u in available.first { $0.uid == u } } ?? available.first
        guard let wanted else { return }
        do {
            try AVAudioSession.sharedInstance().setPreferredInput(wanted)
        } catch {
            note("could not select \(wanted.portName): \(describe(error))")
        }
    }

    /// How much we want each input, highest first.
    ///
    /// A car kit and a pair of AirPods are both `.bluetoothHFP`, and iOS
    /// offers nothing but the name to tell them apart. That makes this a
    /// heuristic, which is worth saying plainly -- but the alternative is
    /// ranking them equally, and a hands-free car microphone is markedly
    /// worse than the phone's own, so getting it wrong in that direction is
    /// the more costly mistake. An unrecognised Bluetooth device therefore
    /// loses to the built-in microphone rather than beating it, and anything
    /// can still be pinned by hand.
    private static func rank(_ p: AVAudioSessionPortDescription) -> Int {
        switch p.portType {
        case .headsetMic:    return 40   // wired, predictable, close to the mouth
        case .bluetoothHFP:  return isAppleEarbuds(p.portName) ? 50 : 5
        case .builtInMic:    return 30
        case .carAudio:      return 0    // never automatically
        default:             return 10
        }
    }

    private static func isAppleEarbuds(_ name: String) -> Bool {
        let n = name.lowercased()
        return n.contains("airpods") || n.contains("beats")
    }

    /// A human name for an input, for the picker and the diagnostics.
    static func label(_ p: AVAudioSessionPortDescription) -> String {
        switch p.portType {
        case .builtInMic:   return "\(p.portName) (built in)"
        case .carAudio:     return "\(p.portName) (car)"
        case .bluetoothHFP: return "\(p.portName) (bluetooth)"
        case .headsetMic:   return "\(p.portName) (wired)"
        default:            return p.portName
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

    private static var configured = false
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
            // Everything is gone, including the category.
            configured = false
            try? ensureActive()
            NotificationCenter.default.post(name: .syrinxAudioReset, object: nil)
        }
        centre.addObserver(forName: AVAudioSession.routeChangeNotification,
                           object: nil, queue: .main) { note in
            let raw = note.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt ?? 0
            let reason = AVAudioSession.RouteChangeReason(rawValue: raw)
            record("route change (\(reason.map(String.init(describing:)) ?? "?"))")
            // The engine holds a tap built for the old input's format, which
            // the new one will not match. Whoever owns it has to rebuild.
            NotificationCenter.default.post(name: .syrinxRouteChanged, object: nil)
        }
    }

    /// Record something worth knowing about later. Visible in Settings.
    static func note(_ what: String) { record(what) }

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
        category: \(s.category.rawValue.replacingOccurrences(of: "AVAudioSessionCategory", with: "")) / \(s.mode.rawValue.replacingOccurrences(of: "AVAudioSessionMode", with: ""))
        options: \(s.categoryOptions.rawValue)
        mic permission: \(permission)
        other audio: \(s.isOtherAudioPlaying)
        input: \(currentInput.map(label) ?? "none")
        available: \(inputs.map(\.portName).joined(separator: ", "))
        configured: \(configured)
        last event: \(lastEvent) (\(eventCount) total)
        """
    }
}

extension Notification.Name {
    /// Everything audio has to be rebuilt.
    static let syrinxAudioReset = Notification.Name("space.aragonite.syrinx.audioReset")
    /// The input changed; the engine's tap no longer matches it.
    static let syrinxRouteChanged = Notification.Name("space.aragonite.syrinx.routeChanged")
}
