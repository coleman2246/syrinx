import AVFoundation
import Foundation

/// Keeps the app resident so the keyboard can start dictation on its own.
///
/// The keyboard cannot launch the app: an extension has no way to open a URL,
/// and iOS has no way to launch an app into the background even if it did. A
/// launch would foreground Syrinx and throw the user out of the field they are
/// typing in, which is worse than asking them to open it. So instead of
/// starting the app on demand, the app stops being suspended.
///
/// iOS suspends a backgrounded app within about thirty seconds unless it is
/// doing something the system treats as ongoing. Audio counts. Recording would
/// also count, but holding the microphone open purely to stay alive lights the
/// orange indicator and misrepresents what the app is doing. Playing silence
/// consumes nothing, shows no indicator, and is honest about both.
///
/// `.mixWithOthers` because this must never interrupt music or a podcast --
/// there is no reason for anything to duck for a track of silence.
final class KeepAwake {
    static let shared = KeepAwake()

    private var player: AVAudioPlayer?
    private var observer: NSObjectProtocol?

    var isRunning: Bool { player?.isPlaying ?? false }

    func start() {
        guard player == nil else { return }
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
            try session.setActive(true)
            let p = try AVAudioPlayer(data: Self.silence())
            p.numberOfLoops = -1
            p.volume = 0
            p.play()
            player = p
        } catch {
            player = nil
            return
        }

        // A phone call or an app that demands the route exclusively will stop
        // playback, and a stopped player is a suspendable app. Resume once the
        // interruption is over, or being interrupted once undoes this quietly.
        observer = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: AVAudioSession.sharedInstance(),
            queue: .main
        ) { [weak self] note in
            guard let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
                  AVAudioSession.InterruptionType(rawValue: raw) == .ended
            else { return }
            self?.restart()
        }
    }

    func stop() {
        player?.stop()
        player = nil
        if let o = observer {
            NotificationCenter.default.removeObserver(o)
            observer = nil
        }
    }

    private func restart() {
        stop()
        start()
    }

    /// A one-second silent WAV, built rather than shipped.
    ///
    /// A file in the bundle would do the same job, but a binary asset nobody
    /// can read is a worse thing to keep in a repository than the twenty lines
    /// that describe exactly what it contains.
    private static func silence(seconds: Double = 1.0, rate: Int = 8000) -> Data {
        let frames = Int(Double(rate) * seconds)
        let payload = frames * 2  // 16-bit mono
        func u32(_ v: Int) -> Data { withUnsafeBytes(of: UInt32(v).littleEndian) { Data($0) } }
        func u16(_ v: Int) -> Data { withUnsafeBytes(of: UInt16(v).littleEndian) { Data($0) } }

        var d = Data("RIFF".utf8)
        d += u32(36 + payload)
        d += Data("WAVEfmt ".utf8)
        d += u32(16)            // PCM header length
        d += u16(1)             // PCM, uncompressed
        d += u16(1)             // mono
        d += u32(rate)
        d += u32(rate * 2)      // bytes per second
        d += u16(2)             // block align
        d += u16(16)            // bits per sample
        d += Data("data".utf8)
        d += u32(payload)
        d += Data(count: payload)
        return d
    }
}
