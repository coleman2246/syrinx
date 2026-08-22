import AVFoundation

/// The one owner of the process's audio session.
///
/// iOS refuses to *activate* an audio session from the background. Category
/// options change which refusals you get, not whether you get one: the error
/// is '!int', and both a non-mixable `.record` session and a mixable
/// `.playAndRecord` one hit it, because activation is what is gated.
///
/// So activation never happens in the background. The session is brought up
/// once at launch, while the app is still on screen and allowed to, and then
/// held for the life of the process. Starting capture later is only starting
/// an engine on a session that is already live, which iOS does permit.
///
/// The cost is real and worth stating: a `.playAndRecord` session held open
/// means the microphone indicator stays lit whenever the app is resident, even
/// with nothing being recorded. That is the price of the keyboard being able
/// to start dictation at all, and it is what the "Keep Syrinx awake" switch
/// actually buys.
enum AudioSession {
    /// Tracked rather than asked for: `AVAudioSession` exposes no `isActive`,
    /// and activating an already-active session from the background is exactly
    /// the call that fails.
    private(set) static var isActive = false

    /// Bring the session up, or confirm it is already up. Safe to call again.
    static func ensureActive() throws {
        let s = AVAudioSession.sharedInstance()
        if s.category != .playAndRecord || !s.categoryOptions.contains(.mixWithOthers) {
            // .measurement disables the input processing that would otherwise
            // fight the recogniser. .mixWithOthers is not accepted on .record,
            // which is why the category is .playAndRecord despite nothing ever
            // being played through it.
            try s.setCategory(.playAndRecord, mode: .measurement,
                              options: [.mixWithOthers, .allowBluetooth])
        }
        guard !isActive else { return }
        try s.setActive(true, options: [])
        isActive = true
    }

    /// Give the session up, which also clears the microphone indicator.
    static func deactivate() {
        guard isActive else { return }
        try? AVAudioSession.sharedInstance()
            .setActive(false, options: [.notifyOthersOnDeactivation])
        isActive = false
    }
}
