import AVFoundation

/// Microphone capture, resampled to what the core wants.
///
/// This is the part that genuinely has to be native. Everything downstream of
/// the callback is Rust.
///
/// The hardware format is not negotiable — the input node reports whatever the
/// device is doing, typically 48 kHz stereo — so a converter sits between it
/// and the 16 kHz mono the model expects. Getting this wrong does not fail
/// loudly; it transcribes gibberish, which is much harder to diagnose than a
/// crash.
final class AudioCapture {
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private let target: AVAudioFormat

    /// Called on the audio thread with 16 kHz mono samples. Must not block.
    private let onSamples: (UnsafePointer<Float>, Int) -> Void

    init?(onSamples: @escaping (UnsafePointer<Float>, Int) -> Void) {
        guard let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: SyrinxCoreSession.sampleRate,
            channels: 1,
            interleaved: false
        ) else { return nil }
        self.target = fmt
        self.onSamples = onSamples
    }

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        // .playAndRecord even though nothing is ever played.
        //
        // The category has to be mixable, because iOS will not let a
        // background app activate a non-mixable session -- it returns '!int',
        // which is what the keyboard's start button kept hitting. And
        // .mixWithOthers is only valid on .playback, .playAndRecord and
        // .multiRoute: setting it on .record, which is what this used to do,
        // is silently ignored and leaves the session non-mixable.
        //
        // So the category is chosen for the option it admits rather than for
        // the direction audio travels. .measurement still disables the input
        // processing that would otherwise fight the recogniser.
        try session.setCategory(.playAndRecord, mode: .measurement,
                                options: [.mixWithOthers, .allowBluetooth])
        try session.setActive(true, options: [])

        let input = engine.inputNode
        let hardware = input.outputFormat(forBus: 0)
        guard hardware.sampleRate > 0 else {
            throw NSError(domain: "Syrinx", code: 1, userInfo: [
                NSLocalizedDescriptionKey:
                    "the microphone reported no format — is another app holding it?"
            ])
        }
        converter = AVAudioConverter(from: hardware, to: target)

        let ratio = target.sampleRate / hardware.sampleRate
        input.installTap(onBus: 0, bufferSize: 4096, format: hardware) { [weak self] buf, _ in
            guard let self, let converter = self.converter else { return }

            // Capacity must be computed from the ratio, rounded up: too small
            // and the converter silently truncates.
            let capacity = AVAudioFrameCount((Double(buf.frameLength) * ratio).rounded(.up)) + 1
            guard let out = AVAudioPCMBuffer(pcmFormat: self.target, frameCapacity: capacity)
            else { return }

            var pushed = false
            var err: NSError?
            converter.convert(to: out, error: &err) { _, status in
                // The converter asks repeatedly; hand over the input once and
                // then report endOfStream, or it spins.
                if pushed {
                    status.pointee = .noDataNow
                    return nil
                }
                pushed = true
                status.pointee = .haveData
                return buf
            }
            if err != nil { return }

            guard out.frameLength > 0, let ch = out.floatChannelData?[0] else { return }
            self.onSamples(ch, Int(out.frameLength))
        }

        engine.prepare()
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        converter = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
    }

    /// Turn an audio error into something a person can act on.
    ///
    /// CoreAudio reports failures as four-character codes rendered in decimal,
    /// so a user sees "OSStatus error 560557684" where the code actually reads
    /// '!int'. The number is unsearchable and the message that wraps it says
    /// only that the operation could not be completed.
    static func describe(_ error: Error) -> String {
        let ns = error as NSError
        guard let code = fourCharCode(ns.code) else { return error.localizedDescription }
        switch code {
        case "!int":
            return "iOS would not start the microphone while Syrinx was in the "
                + "background (\'!int\'). Open the app and start from there."
        case "!pri":
            return "Another app is holding the microphone (\'!pri\')."
        case "!ini":
            return "The audio session was not ready (\'!ini\'). Try again."
        case "!dev":
            return "No microphone is available (\'!dev\')."
        default:
            return "\(error.localizedDescription) (\'\(code)\')"
        }
    }

    /// The printable four-character code inside an OSStatus, if it is one.
    private static func fourCharCode(_ code: Int) -> String? {
        let v = UInt32(bitPattern: Int32(truncatingIfNeeded: code))
        let bytes = [UInt8(truncatingIfNeeded: v >> 24), UInt8(truncatingIfNeeded: v >> 16),
                     UInt8(truncatingIfNeeded: v >> 8), UInt8(truncatingIfNeeded: v)]
        guard bytes.allSatisfy({ (0x20...0x7e).contains($0) }) else { return nil }
        return String(bytes: bytes, encoding: .ascii)
    }

    /// Ask for microphone permission. Without this the tap delivers silence
    /// rather than failing, which looks exactly like a broken server.
    static func requestPermission(_ done: @escaping (Bool) -> Void) {
        if #available(iOS 17.0, *) {
            AVAudioApplication.requestRecordPermission { ok in
                DispatchQueue.main.async { done(ok) }
            }
        } else {
            AVAudioSession.sharedInstance().requestRecordPermission { ok in
                DispatchQueue.main.async { done(ok) }
            }
        }
    }
}
